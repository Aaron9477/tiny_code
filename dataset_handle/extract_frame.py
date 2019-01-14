#!/usr/bin/env python
#coding=utf-8

##---extract frames from videos ---###
# 该文件已用处不大,现在直接从视频中跟踪得到数据集

import os
import cv2
# import cv
import utils

source_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_videos"
output_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_frames"
# intervel_frame = 30 # the intarvel of choosing a frame

if __name__ == "__main__":
    # videos = os.listdir(source_dir)
    # videos = filter(lambda x: x.endswith('.mp4'), videos)

    # videos = ['turtlebot2_7.mp4', 'turtlebot3_7.mp4']
    videos = ['two_small_1.mp4', 'two_small_2.mp4', 'two_small_3.mp4', 'two_small_4.mp4']

    for each_video in videos:
        print('start process {}'.format(each_video))
        each_video_name, _ = each_video.split('.')

        # get the dir of source files and target saving folder
        each_video_file = os.path.join(source_dir, each_video)
        each_video_frame_folder_dir = os.path.join(output_dir, each_video_name)
        if not os.path.exists(each_video_frame_folder_dir):
            utils.build_dir(each_video_frame_folder_dir)
        else:
            continue
        
        cap = cv2.VideoCapture(each_video_file)
        frame_count = 0
        success = True
        while(success):
            success, frame = cap.read()
            params = [1, 100]   # 1 means save jpeg, 100 means the qulity of picture
            cv2.imwrite(os.path.join(each_video_frame_folder_dir, each_video_name+"_{}.jpg".format(frame_count)), frame, params)
            frame_count += 1
        print('{} finished processing!'.format(each_video_name))
    cap.release()


