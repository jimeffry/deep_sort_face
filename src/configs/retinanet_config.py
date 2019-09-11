import os
import sys
from easydict import EasyDict
cfgs = EasyDict()

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
#************************************************************dataset
cfgs.ClsNum = 8
#cfgs.COCODataNames = ['person','bicycle','motorcycle','car','bus'] #'airplane','train','boat']
cfgs.VOCDataNames = ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']
#cfgs.VOCDataNames = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
 #   'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
cfgs.PIXEL_MEAN = [0.485,0.456,0.406] # R, G, B
cfgs.PIXEL_NORM = [0.229,0.224,0.225] #rgb
cfgs.variance = [0.1, 0.2]
cfgs.voc_file = '/home/lxy/Develop/git_prj/refinedet-pytorch/datas/VOC/test_voc07.txt'  
cfgs.voc_dir = '/data/VOC/VOCdevkit'
cfgs.coco_file = '/home/lixiaoyu/Develop/retinanet-pytorch/data/train_coco.txt'
cfgs.coco_dir = '/wdc/LXY.data/CoCo2017'
#**********************************************************************train
cfgs.Show_train_info = 100
cfgs.Smry_iter = 2000
cfgs.Total_Imgs = 133459#133644
cfgs.ImgSize = 320
cfgs.Pkl_Path = '/data/train_record/voc_coco.pkl'
cfgs.ModelPrefix = 'coco_retinanet' #'coco_retinanet' #'coco_resnet_50_state_dict' #
cfgs.Momentum = 0.9
cfgs.Weight_decay = 5e-4
cfgs.lr_steps = [20000, 40000, 60000]
cfgs.lr_gamma = 0.1
cfgs.epoch_num = 120000
#*******************************************************test
cfgs.top_k = 300
cfgs.score_threshold = 0.5
cfgs.nms_threshold = 0.45 #0.45
cfgs.model_dir = '/data/models/retinanet'

# cfgs.shownames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                 'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
#                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                 'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
#                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                 'teddy bear', 'hair drier', 'toothbrush']
cfgs.shownames = ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']