#!/usr/bin/env python

##----根据已经提取的roi从图片中提取相应部分---##
## 这个程序自己电脑跑不了，因为txt文件中的路径不对

import os
import cv2

IMG_SAVE_PATH = '/home/zq610/WYZ/JD_contest/roi'   # 抠出来的图存储路径
DATA_PATH = '/home/zq610/WYZ/JD_contest/detect_txt_all'             # txt文件路径
PREFIX = 'comp4_det_test_'  # txt文件前缀
SUFFIX = '.txt'             # txt文件后缀
FILES = ['cow', 'dog', 'elephant', 'horse', 'sheep']
THRESHOLD = 0.25            # 只抠概率大于0.25的图
SIZE = (640, 360)           # 缩放的size

DIC = {}


# 根据文件名获取到这个图片下一个可以存储的路径名称，因为同一张图片可以抠出多张图
def get_save_path(img_path):
    img_index = str(img_path).strip('\n').split('/')[-1].split('.')[0]
    if not DIC.__contains__(img_index):
        DIC[img_index] = img_save_index = 0
    else:
        DIC[img_index] = img_save_index = DIC[img_index] + 1
    return os.path.join(IMG_SAVE_PATH, img_index + '_' + str(img_save_index) + '.jpg')


# 提取一个图片，x1-4分别为？？？？
def extract_img(img, x1,x2,x3,x4):
    if not os.path.exists(img):
        print(img, ' not found...')
        return
    im = cv2.resize(cv2.imread(img),SIZE, interpolation=cv2.INTER_CUBIC)
    im_save = im[x1:x3, x2:x4]
    # print(get_save_path(img))
    cv2.imwrite(get_save_path(img), im_save)


# 抠出一个txt文件中所有图
def extract_imgs(file):
    f = open(file)
    for line in f.readlines():
        line = line.strip('\n')
        params = line.split(' ')
        if float(params[1]) > THRESHOLD:
            extract_img(params[0], int(float(params[2])), int(float(params[3])),
                        int(float(params[4])), int(float(params[5])))


if __name__ == '__main__':
    if not os.path.exists(IMG_SAVE_PATH):
        os.mkdir(IMG_SAVE_PATH)
    for i in FILES:
        extract_imgs(os.path.join(DATA_PATH, PREFIX + i + SUFFIX))