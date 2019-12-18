#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from head_detection import Detect
from config_head import cfg
from s3fd import build_s3fd
from torch.autograd import Variable

def parms():
    parser = argparse.ArgumentParser(description='s3df demo')
    parser.add_argument('--save_dir', type=str, default='tmp/',
                        help='Directory for detect result')
    parser.add_argument('--headmodelpath', type=str,
                        default='weights/s3fd.pth', help='trained model')
    parser.add_argument('--conf_thresh', default=0.65, type=float,
                        help='Final confidence threshold')
    parser.add_argument('--ctx', default=True, type=bool,
                        help='gpu run')
    parser.add_argument('--img_dir', type=str, default='tmp/',
                        help='Directory for images')
    parser.add_argument('--file_in', type=str, default='tmp.txt',
                        help='image namesf')
    return parser.parse_args()


class HeadDetect(object):
    def __init__(self,args):
        if args.ctx and torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        self.loadmodel(args.headmodelpath)
        self.threshold = args.conf_thresh
        self.img_dir = args.img_dir
        
        self.detect = Detect(cfg)

    def loadmodel(self,modelpath):
        if self.use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.net = build_s3fd('test', cfg.NUM_CLASSES)
        self.net.load_state_dict(torch.load(modelpath,map_location=device))
        self.net.eval()
        if self.use_cuda:
            self.net.cuda()
            cudnn.benckmark = True
    def get_hotmaps(self,conf_maps):
        '''
        conf_maps: feature_pyramid maps for classification
        '''
        hotmaps = []
        print('feature maps num:',len(conf_maps))
        for tmp_map in conf_maps:
            batch,h,w,c = tmp_map.size()
            tmp_map = tmp_map.view(batch,h,w,-1,self.num_classes)
            tmp_map = tmp_map[0,:,:,:,1:]
            tmp_map_soft = torch.nn.functional.softmax(tmp_map,dim=3)
            cls_mask = torch.argmax(tmp_map_soft,dim=3,keepdim=True)
            #score,cls_mask = torch.max(tmp_map_soft,dim=4,keepdim=True)
            #cls_mask = cls_mask.unsqueeze(4).expand_as(tmp_map_soft)
            #print(cls_mask.data.size(),tmp_map_soft.data.size())
            tmp_hotmap = tmp_map_soft.gather(3,cls_mask)
            map_mask = torch.argmax(tmp_hotmap,dim=2,keepdim=True)
            tmp_hotmap = tmp_hotmap.gather(2,map_mask)
            tmp_hotmap.squeeze_(3)
            tmp_hotmap.squeeze_(2)
            print('map max:',tmp_hotmap.data.max())
            hotmaps.append(tmp_hotmap.data.numpy())
        return hotmaps
    def display_hotmap(self,hotmaps):
        '''
        hotmaps: a list of hot map ,every shape is [1,h,w]
        '''       
        row_num = 2
        col_num = 3
        fig, axes = plt.subplots(nrows=row_num, ncols=col_num, constrained_layout=True)
        for i in range(row_num):
            for j in range(col_num):
                #ax_name = 'ax_%s' % (str(i*col_num+j))
                #im_name = 'im_%s' % (str(i*col_num+j))
                ax_name = axes[i,j]
                im_name = ax_name.imshow(hotmaps[i*col_num+j])
                ax_name.set_title("feature_%d" %(i*col_num+j+3))
        #**************************************************************
        img = hotmaps[-1]
        min_d = np.min(img)
        max_d = np.max(img)
        tick_d = []
        while min_d < max_d:
            tick_d.append(min_d)
            min_d+=0.01
        cb4 = fig.colorbar(im_name) #ticks=tick_d)
        plt.savefig('hotmap.png')
        plt.show()
    def propress(self,img):
        rgb_mean = np.array([123.,117.,104.])[np.newaxis, np.newaxis,:].astype('float32')
        img = cv2.resize(img,(cfg.resize_width,cfg.resize_height))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype('float32')
        img -= rgb_mean
        #img = img[:,:,::-1]
        img = np.transpose(img,(2,0,1))
        return img
    def xyxy2xywh(self,bbox_score,scale):
        bboxes = bbox_score[0]
        bbox = bboxes[0] * scale
        score = bboxes[1]
        bbox[:,2] = bbox[:,2] -bbox[:,0] 
        bbox[:,3] = bbox[:,3] -bbox[:,1]  
        return bbox,score
    def inference_img(self,imgorg):
        t1 = time.time()
        imgh,imgw = imgorg.shape[:2]
        img = self.propress(imgorg.copy())
        bt_img = Variable(torch.from_numpy(img).unsqueeze(0))
        if self.use_cuda:
            bt_img = bt_img.cuda()
        output = self.net(bt_img)
        t2 = time.time()
        with torch.no_grad():
            bboxes = self.detect.forward(output[0],output[1],output[2])
        t3 = time.time()
        print('consuming:',t2-t1,t3-t2)
        #showimg = self.label_show(bboxes,imgorg)
        bbox = []
        score = []
        scale = np.array([imgw,imgh,imgw,imgh])
        scale = np.expand_dims(scale,0)
        if len(bboxes)>0:
            bbox,score = self.xyxy2xywh(bboxes,scale)
        # showimg = self.label_show(bbox,score,imgorg)
        return bbox,score
    def label_show(self,rectangles,scores,img):
        imgh,imgw,_ = img.shape
        scale = np.array([imgw,imgh,imgw,imgh])
        # for i in range(len(rectangles)):
        #     bbox = rectangles[i][0]
        #     scores = rectangles[i][1]
        #     for j in range(bbox.shape[0]):
        #         # tmp = np.array(tmp)
        #         score = scores[j]
        #         dets = bbox[j] * scale
        #         x1,y1,x2,y2 = dets
        #         cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
        #         txt = "{:.3f}".format(score)
        #         point = (int(x1),int(y1-5))
        #         #cv2.putText(img,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        for j in range(rectangles.shape[0]):
            dets = rectangles[j]
            score = scores[j]
            x1,y1 = dets[:2]
            x2,y2 = dets[:2] +dets[2:]
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            txt = "{:.3f}".format(score)
            point = (int(x1),int(y1-5))
            cv2.putText(img,txt,point,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
        return img

    def detectheads(self,imgpath):
        if os.path.isdir(imgpath):
            cnts = os.listdir(imgpath)
            for tmp in cnts:
                tmppath = os.path.join(imgpath,tmp.strip())
                img = cv2.imread(tmppath)
                if img is None:
                    continue
                showimg,_ = self.inference_img(img)
                cv2.imshow('demo',showimg)
                cv2.waitKey(0)
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            # if not os.path.exists(self.save_dir):
            #     os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                if len(tmp_file.split(','))>0:
                    tmp_file = tmp_file.split(',')[0]
                if not tmp_file.endswith('jpg'):
                    tmp_file = tmp_file +'.jpeg'
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                if not os.path.exists(tmp_path):
                    print(tmp_path)
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                frame,_ = self.inference_img(img)                
                cv2.imshow('result',frame)
                #savepath = os.path.join(self.save_dir,save_name)
                #cv2.imwrite('test.jpg',frame)
                cv2.waitKey(0) 
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            cap = cv2.VideoCapture(imgpath)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,_ = self.inference_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,odm_maps = self.inference_img(img)
                # hotmaps = self.get_hotmaps(odm_maps)
                # self.display_hotmap(hotmaps)
                # keybindings for display
                cv2.imshow('result',frame)
                #cv2.imwrite('test30.jpg',frame)
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    detector = HeadDetect(args)
    imgpath = args.file_in
    detector.detectheads(imgpath)