#!/usr/bin/env python

#----wipe out the images extracted from same image, and leave only one---##


import os
import cv2

src_dir = '/home/zq610/WYZ/JD_contest/wipe_out/good'
output_dir = '/home/zq610/WYZ/JD_contest/wipe_out/one_left2'

def build_dir(path):
    if not os.path.exists(path):    # build new folder
        os.makedirs(path)

def copyFiles(sourceFile,  targetFile):  # copy
    if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):  
                    open(targetFile, "wb").write(open(sourceFile, "rb").read())

# return the size of image
def get_img_size(input_dir, image_name):
    tmp_img_dir = os.path.join(input_dir, image_name)    #get img's dir
    tmp_img_info = cv2.imread(tmp_img_dir)
    tmp_img_shape = tmp_img_info.shape
    tmp_img_size = tmp_img_shape[0] * tmp_img_shape[1]
    return tmp_img_size

# return the largest img in list
def get_largest_img(input_dir, img_list):
    img_size = 0
    for tmp_img in img_list:
        tmp_img_size = get_img_size(input_dir, tmp_img)
        if tmp_img_size > img_size:
            largest_image = tmp_img
    return largest_image

def main():
    for pig in range(1,31):
        good_dir = os.path.join(output_dir, 'good', str(pig))
        bad_dir = os.path.join(output_dir, 'bad', str(pig))
        build_dir(good_dir)
        build_dir(bad_dir)

        pig_dir = os.path.join(src_dir, str(pig))
        image_list = os.listdir(pig_dir)   #all images
        this_picture_images = []  # save the image from same picture as tmp
        largest_image_list = []
        for frame in range(1,297):  # traverse every picture, one picture could generate many images
            for image in image_list:    # traverse all images to find images from this picture
                if image.startswith(str(frame)+'_'):  #the image start with seqence and '_'
                    this_picture_images.append(image)
            if len(this_picture_images) == 0:
                continue
            elif len(this_picture_images) == 1:
                target_img = this_picture_images[0]
                # image_list.remove(image)    # remove the picture from image_list to reduce the time of computation
            else:
                target_img = get_largest_img(pig_dir, this_picture_images)
            largest_image_list.append(target_img)
            copyFiles(os.path.join(pig_dir, target_img), os.path.join(good_dir, target_img))
            this_picture_images.remove(target_img)  # confirm good frames dont have bad
            # image_list.remove(target_img)  # remove the picture from image_list to reduce the time of computation
            for bad_image in this_picture_images:
                copyFiles(os.path.join(pig_dir, bad_image), os.path.join(bad_dir, bad_image))
                # image_list.remove(bad_image)    # remove the picture from image_list to reduce the time of computation
            print('this list consists ' + str(this_picture_images))
            this_picture_images = []
            target_img = None
            print(str(frame) + ' frame finished')
        print(largest_image_list)

if __name__ == '__main__':
    main()





