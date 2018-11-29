#!/usr/bin/env python
#coding=utf-8

##--- choose one picture in the dataset with a interval ---###

import os
import cv2
# import cv
import utils
# tqdm for progress visualization
from tqdm import tqdm

source_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/raw_frames"
output_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/JPEGImages"
interval = 20

if __name__ == '__main__':

    utils.build_dir(output_dir)
    index = 0

    folders = os.listdir(source_dir)
    # folders = ['two_small_1', 'two_small_2', 'two_small_3', 'two_small_4', ]
    for folder in folders:
        pic_source = os.path.join(source_dir, folder)
        pics = os.listdir(pic_source)

        # sorted不改变原list,只是返回一个list
        # key可以使用lambda表达式,这个x是形参,代表每一个pics中的元素
        pics.sort(key=lambda x:int(x.split('.')[0].split('_')[2]))
        # pics = sorted(pics, key=lambda pics:int(pics.split('.')[0].split('_')[2]))

        count = 0
        for tmp_index in tqdm(range(0, len(pics), interval)):
            utils.copyFiles(os.path.join(pic_source, pics[tmp_index]), os.path.join(output_dir, '%06d.jpg'%(index)))
            count += 1
            index += 1
            # print(pics[tmp_index])
        print('\n{a} finished, there are {b} pictures chosen'.format(a=folder, b=count))


