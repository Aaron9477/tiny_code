#!/usr/bin/env python

##----wipe out the images whose width or height is too small---##

import os
import cv2

src_dir = '/home/zq610/WYZ/JD_contest/wipe_out/one_left/good'
output_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small'
single_threshold = 150  #if height or withd is smaller than it, it will be recongized as bad images
double_threshold = 200  #if height and withd are smaller than it, it will be recongized as bad images

def build_dir(path):
    if not os.path.exists(path):    # build new folder
        os.makedirs(path)

def copyFiles(sourceFile,  targetFile):  # copy
    if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):  
                    open(targetFile, "wb").write(open(sourceFile, "rb").read()) 

def main():
    # global src_dir, output_dir, single_threshold, double_threshold
    for pig in range(1,31):
        print(str(pig) + ' starts')
        good_dir = os.path.join(output_dir, 'output', str(pig))
        bad_dir = os.path.join(output_dir, 'small', str(pig))
        build_dir(good_dir)
        build_dir(bad_dir)

        image_list = os.listdir(os.path.join(src_dir, str(pig)))   #all images
        # print(image_list)
        for pig_image in image_list:
            img_dir = os.path.join(src_dir, str(pig), pig_image)
            # print(img_dir)
            img = cv2.imread(img_dir)
            # print(img)
            img_shape = img.shape
            if img_shape[0]<single_threshold or img_shape[1]<single_threshold or (img_shape[0]<double_threshold and img_shape[1]<double_threshold):
                copyFiles(img_dir, os.path.join(bad_dir, pig_image))
                print(pig_image + ' has been cleaned')
            else:
                copyFiles(img_dir, os.path.join(good_dir, pig_image))


if __name__ == '__main__':
    main()

