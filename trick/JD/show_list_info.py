#!/usr/bin/env python

#---show image list information, like the sum of list, the absence of list---#

import os
import cv2

src_dir = '/home/zq610/WYZ/JD_contest/wipe_out/one_left2/good'
num_pig_list = []

def show_num(pig_dir, pig):  #print the num of pigs' images
    this_pig_num = len(os.listdir(pig_dir))
    print('the ' + str(pig) + 'th pig has ' + str(this_pig_num) + ' pictures')
    return this_pig_num


def show_absence(pig_dir, pig):
    image_list = os.listdir(pig_dir)   #all images
    frame_list = range(1,297)
    for frame in range(1,297):  # if there is a picture in the folder, it will be removed from the image_list
        for image in image_list:    # traverse all images to find images from this picture
                if image.startswith(str(frame)+'_'):  #the image start with seqence and '_'
                    frame_list.remove(frame)
    output = ''
    for i in frame_list:
        output = output + str(i) + ', '
    output = output[:-2]
    print(str(pig) + 'th pig doesn\'t have ' + output + 'frames')


def main():
    for pig in range(1,31):
        pig_dir = os.path.join(src_dir, str(pig))
        num_pig_list.append(show_num(pig_dir, pig))
        show_absence(pig_dir, pig)
    print('all finished')

if __name__ == '__main__':
    main()
