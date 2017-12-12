#!/usr/bin/env python


import os
import cv2

result_output = '/home/zq610/WYZ/JD_contest/output'
DATA_PATH = '/home/zq610/WYZ/JD_contest/detect_txt_all'
PREFIX = 'comp4_det_test_'
SUFFIX = '.txt'
FILES = ['cow', 'dog', 'elephant', 'horse', 'sheep']
THRESHOLD = 0.25
# SIZE = (640, 360)
save_prob = []
DIC = {}


def record_img(img_path, chance):
    img_index = str(img_path).strip('\n').split('/')[-1].split('.')[0]
    save_prob[img_index][FILES[0]].append(chance)



# def record_img(img, chance):
#     if not os.path.exists(img):
#         print(img, ' not found...')
#         return
#     im = cv2.resize(cv2.imread(img),SIZE, interpolation=cv2.INTER_CUBIC)
#     im_save = im[x1:x3, x2:x4]
#     # print(get_save_path(img))
#     cv2.imwrite(get_save_path(img), im_save)


def extract_imgs(file):
    f = open(file)
    for line in f.readlines():
        line = line.strip('\n')
        params = line.split(' ')
        if float(params[1]) > THRESHOLD:
            record_img(params[0], float(params[1]))
    print(save_prob)



# def mian():




if __name__ == '__main__':
    i = FILES[0]
    extract_imgs(os.path.join(DATA_PATH, PREFIX + i + SUFFIX))