import numpy as np 
from easydict import EasyDict

config = EasyDict()
# print time consuming
config.time = 0
#if crop_org=1, the main file--test.py will directly output the result by giving face detect model
config.crop_org = 0
#if x_y=1, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,y1,x2,y2,...,x5,y5
#if x_y=0, coordinates of the 5 points(eye,nose,mouse) will be ordered x1,x2,x3,x4,x5,y1,y2,y3,y4,y5
config.x_y = 0
#whether to downsample img to short size is 320
config.img_downsample = 0
#whether to keep the original ratio to get ouput size
config.imgpad = 0
# whether to show the picture
config.img_show = 0
#detect face with caffe or mxnet, 0-caffe,1-mxnet
config.caffe_use = 0
#used to adjust face detect range
config.min_size = 50
#if gpu_list =[] codes will run on cpu.
config.gpu_list = []
# select topk proposals according to score in Pnet
config.topk = 10
#batch-size
config.batch_size = 5