from easydict import EasyDict

config = EasyDict()
# if caffe_use=1, face_model.py file will load caffe model to generate face features for face recognition
config.caffe_use = 0
# The value of feature_lenth is 512 or 256. The length of the feature is determined by the dimension of the feature
config.feature_lenth = 512#256#1024
# method of mxnet load model
config.mx_loadmodel = 0
#if debug=1,will pring information among  all the face reg files
config.debug = 0
#only used for mxnet and caffemodel, if feature_expand=1, the input of the mxnet net will be 2 images, and the output features of the face will add the 2 images features. 
config.feature_expand = 0
# the subtraction of  top 2 distances, threshold value
config.confidence = 0.1
# the nearest person, distance threshold value
config.top1_distance = 1.2
# metric distance
config.cosin_distance = 0
# whether to print the detect and recognize consuming time
config.time = 0
# after load caffe models successful, resaving the caffemodel
config.model_resave = 0
# whether to display and save model structure for mxnet
config.display_model = 0
#inference img batches
config.batch_size = 1
#whether to save result about recognize
config.saveresult = 0
#whether to build database by runing face_detect
config.builddb_detect = 0
#if gpu_list =[] codes will run on cpu.
config.gpu_list = []