import os
import sys
import cv2
import time
import argparse
import numpy as np
from Detector_mxnet import MtcnnDetector

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

class Demo(object):
    def __init__(self, args):
        self.args = args
        if args.display:
            cv2.namedWindow("test")#cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        #self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names,use_cuda=args.use_cuda, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        threshold = np.array([0.7,0.8,0.9])
        crop_size = [112,112]
        self.mtcnn =  MtcnnDetector(threshold,crop_size,args.detect_model)
        
    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def draw_bboxes(self,img, bbox, identities=None, offset=(0,0)):
        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            h = y2-y1
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = COLORS_10[id%len(COLORS_10)]
            label = '{}{:d}'.format("", id)
            #t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,1)
            #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            #cv2.putText(img,label,(x1,y1+t_size[1]+4), 0, 1e-2*h, [255,255,255], 1)
            cv2.putText(img,label,(x1,y1+4), 0, 1e-2*h, [255,255,255], 1)
        return img

    def detect(self):
        while self.vdo.grab(): 
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = np.array([im])
            rectangles = self.mtcnn.detectFace(im,True)
            rectangles = rectangles[0]
            if len(rectangles) <1:
                    continue
            outputs = rectangles[:,:4]
            if len(outputs) > 0:
                #outputs = rectangles
                bbox_xyxy = outputs[:,:4]
                identities = outputs[:,-1] #np.zeros(show_box.shape[0]) #
                ori_im = self.draw_bboxes(ori_im, bbox_xyxy, identities)
            cv2.imshow("test", ori_im)
            c = cv2.waitKey(1) & 0xFF
            if c==27 or c==ord('q'):
                break
        self.vdo.release()
        cv2.destroyAllWindows()

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
    parser.add_argument('--tracker_type',type=str,default='mosse')
    return parser.parse_args()


if __name__=="__main__":
    #"demo.avi"
    args = parse_args()
    with Demo(args) as det:
        det.detect()
