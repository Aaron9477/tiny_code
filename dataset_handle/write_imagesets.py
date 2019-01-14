#!/usr/bin/env python
#coding=utf-8

import utils
import os
import random

target_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/turtlebot/ImageSets/Main"
num_pix = 1154
val_percent = 0

def build_val_txt():
    trainval_dir = os.path.join(target_dir, "val.txt")
    f = open(trainval_dir, 'w')
    trainval_list = list(range(1, num_pix+1))
    val_list = random.sample(trainval_list, int(num_pix * val_percent))
    val_list.sort()
    for i in val_list:
        f.write('%06d\n'%(i))
    print("finished building val txt!")
    return trainval_list, val_list

def build_train_txt(trainval_list, val_list):
    trainval_dir = os.path.join(target_dir, "train.txt")
    f = open(trainval_dir, 'w')
    train_list = list(set(trainval_list) ^ set(val_list))    # get different set
    # print(train_list)
    # exit()
    train_list.sort()
    for i in train_list:
        f.write('%06d\n'%(i))
    print("finished building train txt!")

def build_trainval_txt():
    trainval_dir = os.path.join(target_dir, "trainval.txt")
    f = open(trainval_dir, 'w')
    for i in range(1, num_pix+1):
        f.write('%06d\n'%(i))
    print("finished building trainval txt!")

if __name__ == "__main__":
    build_trainval_txt()
    trainval_list, val_list = build_val_txt()
    build_train_txt(trainval_list, val_list)


