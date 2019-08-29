#!/usr/bin/bash
#python demo_deepsort.py /data/pedestrian_2.mp4  
#--save_path /data/test/mot_person
python demo_deepsort.py /data/videos/face.mp4 --detect_model /data/models/mtcnn_mxnet  --feature_model /data/models/face_resnet101_mxnet/modelresave --load_num 0
#face_resnet101_mxnet/modelresave model-y1-test2/model