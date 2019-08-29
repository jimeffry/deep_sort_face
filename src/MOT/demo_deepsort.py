import os
import sys
import cv2
import time
import argparse
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'../obj_detector/pedestrian'))
from detector import YOLOv3
sys.path.append(os.path.join(os.path.dirname(__file__),'../obj_detector/face'))
from Detector_mxnet import MtcnnDetector
sys.path.append(os.path.join(os.path.dirname(__file__),'../sort'))
from deep_sort import DeepSort
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from util import COLORS_10, draw_bboxes
sys.path.append(os.path.join(os.path.dirname(__file__),'../tracker'))
from kalman_filter_track import KalmanFilter
from staple import Staple
from mosse import MOSSE
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from staple_config import StapleConfig

class MOTTracker(object):
    def __init__(self, args):
        self.args = args
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        #self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names,use_cuda=args.use_cuda, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        threshold = np.array([0.7,0.8,0.9])
        crop_size = [112,112]
        self.mtcnn =  MtcnnDetector(threshold,crop_size,args.detect_model)
        #self.deepsort = DeepSort(args.deepsort_checkpoint,use_cuda=args.use_cuda)
        self.deepsort = DeepSort(args.feature_model,args.load_num,use_cuda=args.use_cuda)
        self.kf = KalmanFilter()
        self.meanes_track = []
        self.convariances_track = []
        self.id_cnt_dict = dict()
        if args.tracker_type=='moss':
            self.tracker=MOSSE()
        elif args.tracker_type=='staple':
            self.tracker=Staple(config=StapleConfig())

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if  self.args.save_path:
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path)
            #fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            #self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def track_init(self,bbox_xyah):
        means = []
        convariances = []
        for box in bbox_xyah:
            mean,convariance = self.kf.initiate(box)
            #print(mean)
            means.append(mean)
            convariances.append(convariance)
        self.means_track = means
        self.convariances_track = convariances

    def track_predict(self):
        self.mean_update = []
        self.convariance_update = []
        for mean_tmp, conv_tmp in zip(self.means_track,self.convariances_track):
            mean, convariance = self.kf.predict(mean_tmp,conv_tmp)
            self.mean_update.append(mean)
            self.convariance_update.append(convariance)
        #print('pred:',np.shape(self.mean_update))
        self.means_track = self.mean_update
        self.convariances_track = self.convariance_update
    
    def track_update(self,detect_box):
        self.means_track = []
        self.convariances_track = []
        idx = 0
        for mean_tmp,conv_tmp in zip(self.mean_update,self.convariance_update):
            #detect_tmp = mean_tmp[:4]
            #detect_tmp[:2] = detect_tmp[:2] +1
            detect_tmp = detect_box[idx]
            idx+=1
            mean, convariance = self.kf.update(mean_tmp,conv_tmp,detect_tmp)
            #print('updata:',np.shape(mean))
            self.means_track.append(mean)
            self.convariances_track.append(convariance)
        #print('update:',np.shape(self.means_track))

    def xcycah2xcyc(self,xyah):
        xyah = np.array(xyah)
        xyah = xyah[:,:4]
        w = xyah[:,2] * xyah[:,3]
        h = xyah[:,3]
        xc = xyah[:,0] #+ w/2
        yc = xyah[:,1] #+ h/2
        return np.vstack([xc,yc,w,h]).T

    def xcycah2xyxy(self,xcycah):
        xcycah = np.array(xcycah)
        xcycah = xcycah[:,:4]
        w = xcycah[:,2] * xcycah[:,3]
        h = xcycah[:,3]
        x2 = xcycah[:,0] + w/2
        y2 = xcycah[:,1] + h/2
        x1 = xcycah[:,0] - w/2
        y1 = xcycah[:,1] - h/2
        return np.vstack([x1,y1,x2,y2]).T
    
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

    def xcyc2xcycah(self,bbox_xcycwh):
        bbox_xcycwh = np.array(bbox_xcycwh,dtype=np.float32)
        xc = bbox_xcycwh[:,0] #- bbox_xcycwh[:,2]/2
        yc = bbox_xcycwh[:,1] #- bbox_xcycwh[:,3]/2
        a = bbox_xcycwh[:,2] / bbox_xcycwh[:,3]
        return np.vstack([xc,yc,a,bbox_xcycwh[:,3]]).T

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
            tmp_cnt = self.id_cnt_dict.setdefault(id,0)
            self.id_cnt_dict[id] = tmp_cnt+1
            crop_img = img[y1:y2,x1:x2,:]
            save_dir = os.path.join(self.args.save_path,id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir,id+'_'+str(tmp_cnt)+'.jpg')
            cv2.imwrite(save_path,crop_img)

    def detect(self):
        cnt = 0
        update_fg = True
        detect_fg = True
        total_time = 0
        outputs = []
        while self.vdo.grab(): 
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = np.array([im])
            if cnt % 5 ==0 or detect_fg:
                # bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
                # mask = cls_ids==0
                # bbox_xcycwh = bbox_xcycwh[mask]
                # bbox_xcycwh[:,3:] *= 1.2
                # cls_conf = cls_conf[mask]
                rectangles = self.mtcnn.detectFace(im,True)
                rectangles = rectangles[0]
                if len(rectangles) <1:
                    continue
                bboxes = rectangles[:,:4]
                bbox_xcycwh = self.xyxy2xcyc(bboxes)
                cls_conf = rectangles[:,4]
                update_fg = True
                if bbox_xcycwh is not None:
                    #box_xcycah = self.xcyc2xcycah(bbox_xcycwh)
                    detect_xywh = self.xyxy2xywh(bboxes)
                    #self.track_init(box_xcycah)
                    self.tracker.init(im,detect_xywh)
                    detect_fg = False
                    #self.track_predict()
                    #self.track_update(box_xcycah)
            else:
                #print('************here')
                if bbox_xcycwh is not None:
                    self.track_predict()
                    #show_box = self.xyah2xyxy(self.means_track)
                    bbox_xcycwh = self.xcycah2xcyc(self.means_track)
                    update_fg = False
                    detect_fg = False
                else:
                    detect_fg = True
            if bbox_xcycwh is not None:
                # select class person
                #print('detect_box:',bbox_xcycwh.shape)
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im[0],update_fg)
            end = time.time()
            consume = end-start
            if len(outputs) > 0:
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1] #np.zeros(show_box.shape[0]) #
                ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
                #self.save_track_results(bbox_xyxy,ori_im,identities)
            print("frame: {} time: {}s, fps: {}".format(cnt,consume, 1/(end-start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            #if self.args.save_path:
             #   self.output.write(ori_im)
            cnt +=1
            total_time += consume
            #if cnt ==10:
             #   break
        print("video ave fps: ",cnt/total_time,total_time)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="../obj_detector/pedestrian/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="../../models/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="../obj_detector/pedestrian/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="../../models/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--use_cuda",type=int,default=0)
    parser.add_argument("--load_num",type=int,default=0,help='')
    parser.add_argument("--detect_model",type=str,default="../../models/mxnet_model")
    parser.add_argument("--feature_model",type=str,default="../../models/mxnet")
    return parser.parse_args()


if __name__=="__main__":
    #"demo.avi"
    args = parse_args()
    with MOTTracker(args) as det:
        det.detect()
