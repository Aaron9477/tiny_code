#!/usr/bin/env python

#---show images' information---#

import os
import cv2


src_dir = '/home/zq610/WYZ/JD_contest/wipe_out/good/5'
SUFFIX = '.jpg'
img_name = '124_4.jpg'


def search(path=".", name="1"):
    for item in os.listdir(path):  
        item_path = os.path.join(path, item)  
        if os.path.isdir(item_path):  
            search(item_path, name)  
        elif os.path.isfile(item_path):  
            if name in item:
                print(item_path)
                return search()  
    

if __name__ == '__main__':
    # img_dir = search(src_dir, img_name)
    # print(img_dir)
    img_dir2 = os.path.join(src_dir, img_name)
    print(img_dir2)
    img = cv2.imread(img_dir2)
    cv2.namedWindow('FILE')
    cv2.imshow('FILE', img)
    img_shape = img.shape
    print('the image ' + img_name + ' height' + str(img_shape[0]) + ' width:' + str(img_shape[1]) + '\n')  #img_shape[0] height  img_shape[1] width



# src_dir = '/home/zq610/WYZ/JD_contest/wipe_out/good/5'
# image_list = os.listdir(src_dir)

# for FILE in image_list:
#     img_dir = os.path.join(src_dir, FILE)
#     img = cv2.imread(img_dir)
#     img_shape = img.shape
#     print('the image ' + FILE + ' height' + str(img_shape[0]) + ' width:' + str(img_shape[1]) + '\n')  #img_shape[0] height  img_shape[1] width

    # if img_shape[0]<150 or img_shape[1] <150 or (img_shape[0]<200 and img_shape[1] <200):
    #     cv2.namedWindow(FILE)
    #     cv2.imshow(FILE, img)
    #     cv2.waitKey(400)
    #     print('the image ' + FILE + ' height' + str(img_shape[0]) + ' width:' + str(img_shape[1]) + '\n')  #img_shape[0] height  img_shape[1] width
