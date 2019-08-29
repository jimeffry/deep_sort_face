from easydict import EasyDict
cfgs = EasyDict()
#************************************
#
cfgs.debug = 1
#
cfgs.INFTY_COST = 1e+5
cfgs.det_min_confidence = 0.3
cfgs.det_nms_max_overlap = 1.0
# for feature and Maha distance threshold
cfgs.max_cosine_distance = 0.6
cfgs.feature_max_keep = 100
#for iou distance threshold
cfgs.max_iou_distance=0.7
cfgs.max_age=70
cfgs.confirm_frame_cnt=3