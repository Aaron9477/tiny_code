#!bin/bash

# g++ track_and_crop.cpp -o wqe `pkg-config opencv --cflags --libs`

video_dir='/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_videos/turtlebot2_1.mp4'
save_dir='/home/zq610/WYZ/tiny_code/dataset_handle/output/'

./wqe -v ${video_dir} -s ${save_dir}
