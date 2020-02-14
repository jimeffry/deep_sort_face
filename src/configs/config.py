from easydict import EasyDict
cfgs = EasyDict()
#************************************
#
cfgs.debug = 0
#
cfgs.INFTY_COST = 1e+5
cfgs.det_min_confidence = 0.3
cfgs.det_nms_max_overlap = 1.0
# for feature and Maha distance threshold
cfgs.max_cosine_distance = 0.8
cfgs.feature_max_keep = 100
#for iou distance threshold
cfgs.max_iou_distance=0.7
cfgs.max_age=70
cfgs.confirm_frame_cnt=3
cfgs.inout_point = [0,150]
# if 0, y-axis-door: down2up is in, up2down is out; x-axis-door: right2left is in,or out
# if 1, y-axis-door: up2down is in, or out; x-axis-door: left2right is in,or out
cfgs.inout_type = 1 