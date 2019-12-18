#!/usr/bin/bash
#*********face track
#python demo_deepsort.py /data/videos/face.mp4 --detect_model /data/models/mtcnn_mxnet  --feature_model /data/models/model-y1-test2/model --face_load_num 0   --tracker_type staple --mot_type face
#face_resnet101_mxnet/modelresave model-y1-test2/model
#********************person track
#python demo_deepsort.py /data/videos/pedestrian_2.mp4   --feature_model /data/models/deep_sort/ckpt.t7   --person_load_num 65 --tracker_type mosse --mot_type person
#**************************head track
python demo_deepsort.py /data/videos/test1.mp4 --headmodelpath /data/models/head/sfd_crowedhuman_75000.pth  --feature_model /data/models/deep_sort/ckpt.t7 --face_load_num 0   --tracker_type staple --mot_type head
#******************kalman
#python kalman_mot.py /data/videos/pedestrian_2.mp4  --feature_model /data/models/deep_sort/ckpt.t7   --person_load_num 65  --mot_type person

#********************only track
#python detect_by_track.py /data/videos/face.mp4 --detect_model /data/models/mtcnn_mxnet  --tracker_type dat --mot_type face
#python detect_by_track.py /data/videos/pedestrian_2.mp4  --person_load_num 65 --tracker_type mosse --mot_type person
