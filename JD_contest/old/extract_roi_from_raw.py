#!/usr/bin/env python



import os
import cv2

IMG_SAVE_PATH = '/home/zq610/WYZ/JD_contest/roi'
DATA_PATH = '/home/zq610/WYZ/JD_contest/detect_txt_all'            
PREFIX = 'comp4_det_test_'
SUFFIX = '.txt'           
FILES = ['cow', 'dog', 'elephant', 'horse', 'sheep']
THRESHOLD = 0.25           
# SIZE = (640, 360)          
DIC = {}


def get_save_path(img_path):
    img_index = str(img_path).strip('\n').split('/')[-1].split('.')[0]
    if not DIC.__contains__(img_index):
        DIC[img_index] = img_save_index = 0
    else:
        DIC[img_index] = img_save_index = DIC[img_index] + 1
    return os.path.join(IMG_SAVE_PATH, img_index + '_' + str(img_save_index) + '.jpg')


def extract_img(img, x1,x2,x3,x4):
    if not os.path.exists(img):
        print(img, ' not found...')
        return
    # resize to (640,360)
    # im = cv2.resize(cv2.imread(img),SIZE, interpolation=cv2.INTER_CUBIC)
    # im_save = im[x1:x3, x2:x4]
    # # print(get_save_path(img))
    # cv2.imwrite(get_save_path(img), im_save)
    #no resize
    im = cv2.imread(img)
    im_save = im[x1:x3, x2:x4]
    # print(get_save_path(img))
    cv2.imwrite(get_save_path(img), im_save)


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