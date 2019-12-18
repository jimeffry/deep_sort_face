import os
import sys
import cv2
import time
import argparse
import numpy as np
from tracker_run import TrackerRun,MoveTrackerRun
sys.path.append(os.path.join(os.path.dirname(__file__),'../obj_detector/pedestrian'))
from detector import YOLOv3
sys.path.append(os.path.join(os.path.dirname(__file__),'../obj_detector/retinanet'))
from retinanet_detector import RetinanetDetector
sys.path.append(os.path.join(os.path.dirname(__file__),'../obj_detector/face'))
from Detector_mxnet import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__),'../obj_detector/head'))
from head_detector import HeadDetect
sys.path.append(os.path.join(os.path.dirname(__file__),'../sort'))
from deep_sort import DeepSort
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import COLORS_10, draw_bboxes
from blurdetect import BlurDetection
sys.path.append(os.path.join(os.path.dirname(__file__),'../tracker'))
from kalman_filter_track import KalmanFilter


class MOTTracker(object):
    def __init__(self, args):
        self.args = args
        self.open_video()
        #self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names,use_cuda=args.ctx, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        self.command_type = args.mot_type
        threshold = np.array([0.7,0.8,0.9])
        crop_size = [112,112]
        if self.command_type == 'face':
            self.mtcnn =  MtcnnDetector(threshold,crop_size,args.detect_model)
        elif self.command_type == 'person':
            self.person_detect =  RetinanetDetector(args)
        elif self.command_type == 'head':
            self.head_detect = HeadDetect(args)
        self.kf = KalmanFilter()
        self.deepsort=DeepSort(args.feature_model,args.face_load_num,mot_type=args.mot_type)
        self.meanes_track = []
        self.convariances_track = []
        self.id_cnt_dict = dict()
        self.tracker_run = TrackerRun(args.tracker_type)
        self.moveTrack = MoveTrackerRun(self.kf)
        self.img_clarity = BlurDetection()
        self.score = 60.0
        self.in_num = 0
        self.out_num = 0

    def open_video(self):
        if  not os.path.isfile(self.args.VIDEO_PATH):
            raise Exception("Error:input video path is not exist")
        self.vdo = cv2.VideoCapture(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.args.save_dir:
            if not os.path.exists(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            #fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            #self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))
        if not self.vdo.isOpened():
            raise Exception('open video failed')
    def xyxy2xcyc(self,xywh):
        w = xywh[:,2] - xywh[:,0]
        h = xywh[:,3] - xywh[:,1]
        xc = xywh[:,0] + w/2
        yc = xywh[:,1] + h/2
        return np.vstack([xc,yc,w,h]).T
    def xyxy2xywh(self,xywh):
        w = xywh[:,2] - xywh[:,0]
        h = xywh[:,3] - xywh[:,1]
        return np.vstack([xywh[:,0],xywh[:,1],w,h]).T
    def xywh2xcycwh(self,xywh):
        xywh = np.array(xywh)
        xc = xywh[:,0]+xywh[:,2]/2
        yc = xywh[:,1]+xywh[:,3]/2
        return np.vstack([xc,yc,xywh[:,2],xywh[:,3]]).T
    def widerbox(self,boxes):
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        boxw = x2-x1
        boxh = y2-y1
        x1 = np.maximum(0,x1-0.3*boxw)
        y1 = np.maximum(0,y1-0.3*boxh)
        x2 = np.minimum(self.im_width,x2+0.3*boxw)
        y2 = np.minimum(self.im_height,y2+0.3*boxh)
        return np.vstack([x1,y1,x2,y2]).T
        
    def save_track_results(self,bbox_xyxy,img,identities,offset=[0,0]):
        for i,box in enumerate(bbox_xyxy):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            x1 = min(max(x1,0),self.im_width-1)
            y1 = min(max(y1,0),self.im_height-1)
            x2 = min(max(x2,0),self.im_width-1)
            y2 = min(max(y2,0),self.im_height-1)
            # box text and bar
            id = str(identities[i]) if identities is not None else '0'
            crop_img = img[y1:y2,x1:x2,:]
            if self.img_clarity._blurrDetection(crop_img) > self.score:
                tmp_cnt = self.id_cnt_dict.setdefault(id,0)
                self.id_cnt_dict[id] = tmp_cnt+1
                save_dir = os.path.join(self.args.save_dir,id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join(save_dir,id+'_'+str(tmp_cnt)+'.jpg')
                cv2.imwrite(save_path,crop_img)
            else:
                continue
    def show_bboxs(self,img, bbox, identities=None,fgs=[]):
        imh,imw = img.shape[:2]
        for i,box in enumerate(bbox):
            box = list(map(int,box))
            x1,y1,x2,y2 = box
            h = y2-y1
            tm_fg = fgs[i]
            if tm_fg == 0:
                self.out_num +=1
            elif tm_fg == 1:
                self.in_num +=1
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = COLORS_10[id%len(COLORS_10)]
            label = '{}{:d}'.format("", id)
            #t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,4)
            #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            #cv2.putText(img,label,(x1,y1+t_size[1]+4), 0, 1e-2*h, [255,255,255], 1)
            cv2.putText(img,label,(x1,y1+4), 0, 1e-2*h, [255,255,255], 4)
        txt = "in:{}  out:{}".format(self.in_num,self.out_num)
        cv2.putText(img,txt,(imw-100,20),0,0.5,[0,0,255],2)
        cv2.line(img,(200,0),(200,imh-1),(255,0,0),2,4)
        return img

    def detect(self):
        cnt = 0
        update_fg = True
        detect_fg = True
        total_time = 0
        outputs = []
        while self.vdo.isOpened(): 
            start = time.time()
            _, ori_im = self.vdo.read()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = np.array([im])
            if cnt % 20 ==0 or detect_fg:
                # bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
                # mask = cls_ids==0
                # bbox_xcycwh = bbox_xcycwh[mask]
                # bbox_xcycwh[:,3:] *= 1.2
                # cls_conf = cls_conf[mask]
                if self.command_type == 'face':
                    rectangles = self.mtcnn.detectFace(im,True)
                    rectangles = rectangles[0]
                    if len(rectangles) <1:
                        continue
                    bboxes = rectangles[:,:4]
                    bboxes = self.widerbox(bboxes)
                    #
                    bbox_xcycwh = self.xyxy2xcyc(bboxes)
                    cls_conf = rectangles[:,4]
                elif self.command_type=='person':
                    bboxes,cls_conf = self.person_detect.test_img_org(ori_im)
                    if len(bboxes)==0:
                        continue
                    bbox_xcycwh = self.xywh2xcycwh(bboxes)
                elif self.command_type == 'head':
                    bboxes,cls_conf = self.head_detect.inference_img(ori_im)
                    if len(bboxes)==0:
                        continue
                    bbox_xcycwh = self.xywh2xcycwh(bboxes)
                #outputs = bboxes #self.xywh2xyxy(bboxes)
                update_fg = True
                # box_xcycah = self.xcyc2xcycah(bbox_xcycwh)
                # self.moveTrack.track_init(box_xcycah)
                # self.moveTrack.track_predict()
                # self.moveTrack.track_update(box_xcycah)
                detect_xywh = self.xyxy2xywh(bboxes) if self.command_type=='face' else bboxes
                self.tracker_run.init(ori_im,detect_xywh.tolist())
                detect_fg = False
            else:
                #print('************here')
                if len(bbox_xcycwh) >0 :
                    #self.moveTrack.track_predict()
                    #bbox_xcycwh = self.xcycah2xcyc(self.means_track)
                    #outputs = self.xcycah2xyxy(self.moveTrack.means_track)
                    start1 = time.time()
                    boxes_tmp = self.tracker_run.update(ori_im)
                    bbox_xcycwh = self.xywh2xcycwh(boxes_tmp)
                    end1 = time.time() 
                    print('only tracker time consume:',end1-start1)
                    #outputs = self.xywh2xyxy(boxes_tmp)
                    update_fg = False
                    detect_fg = False
                else:
                    detect_fg = True
            if len(bbox_xcycwh)>0:
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, ori_im,update_fg)
            end = time.time()
            consume = end-start
            if len(outputs) > 0:
                #outputs = rectangles
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,4] #np.zeros(outputs.shape[0]) 
                fgs = outputs[:,-1]
                #ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
                ori_im = self.show_bboxs(ori_im,bbox_xyxy,identities,fgs)
                #self.save_track_results(bbox_xyxy,ori_im,identities)
            print("frame: {} time: {}s, fps: {}".format(cnt,consume, 1/(end-start)))
            cnt+=1
            cv2.imshow("test", ori_im)
            c = cv2.waitKey(1) & 0xFF
            if c==27 or c==ord('q'):
                break
            #if self.args.save_path:
             #   self.output.write(ori_im)
            total_time += consume
        self.vdo.release()
        cv2.destroyAllWindows()
        print("video ave fps and total_time: ",cnt/total_time,total_time)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="../obj_detector/pedestrian/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="../../models/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="../obj_detector/pedestrian/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--img_dir",type=str,default='',help='')
    parser.add_argument("--ctx",type=int,default=0)
    parser.add_argument("--face_load_num",type=int,default=0,help='')
    parser.add_argument("--person_load_num",type=int,default=0,help='')
    parser.add_argument("--detect_model",type=str,default="../../models/mxnet_model")
    parser.add_argument("--feature_model",type=str,default="../../models/mxnet")
    parser.add_argument("--headmodelpath",type=str,default="../../models/head")
    parser.add_argument('--tracker_type',type=str,default='mosse')
    parser.add_argument('--mot_type',type=str,default='face',help='face or person')
    return parser.parse_args()


if __name__=="__main__":
    #"demo.avi"
    args = parse_args()
    det = MOTTracker(args)
    det.detect()
