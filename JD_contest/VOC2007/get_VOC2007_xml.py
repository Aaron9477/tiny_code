#!/usr/bin/env python


import os
import utils as u


image_dir = '/home/zq610/WYZ/JD_contest/wipe_out/filling'
txt_dir = '/media/zq610/新加卷/fangxing'
output_dir = ''
folder = ['tian0', 'tian01', 'tian02']
img_SUFFIX = '.jpg'
txt_SUFFIX = '.txt'
xml_SUFFIX = '.xml'


def main():
    this_txt_dir = os.path.join(txt_dir, folder[0])
    for pig in range(1,31):
        this_pig_dir = os.path.join(image_dir, str(pig))
        for FILE in os.listdir(this_pig_dir):
            portion = os.path.splitext(FILE)
            print(portion)
            txt_name = portion[0] + txt_SUFFIX
            print(txt_name)
            u.search(txt_name)
            exit()



        # good_dir = os.path.join(output_dir, 'good', str(pig))
        # bad_dir = os.path.join(output_dir, 'bad', str(pig))
        # build_dir(good_dir)
        # build_dir(bad_dir)
        #
        # pig_dir = os.path.join(src_dir, str(pig))
        # image_list = os.listdir(pig_dir)   #all images
        # this_picture_images = []  # save the image from same picture as tmp
        # largest_image_list = []
        # for frame in range(1, num_picture+1):  # traverse every picture, one picture could generate many images
        #     for image in image_list:    # traverse all images to find images from this picture
        #         if image.startswith(str(pig)+'_'+str(frame)+'_'):  #the image start with seqence and '_'
        #             this_picture_images.append(image)
        #     if len(this_picture_images) == 0:
        #         continue
        #     elif len(this_picture_images) == 1:
        #         target_img = this_picture_images[0]
        #     else:
        #         target_img = get_largest_img(pig_dir, this_picture_images)
        #     largest_image_list.append(target_img)
        #     copyFiles(os.path.join(pig_dir, target_img), os.path.join(good_dir, target_img))
        #     this_picture_images.remove(target_img)  # confirm good frames dont have bad
        #     image_list.remove(target_img)  # remove the picture from image_list to reduce the time of computation
        #     if len(this_picture_images) > 0:
        #         for bad_image in this_picture_images:
        #             copyFiles(os.path.join(pig_dir, bad_image), os.path.join(bad_dir, bad_image))
        #             image_list.remove(bad_image)    # remove the picture from image_list to reduce the time of computation
        #     print('this list consists bad pictures of ' + str(this_picture_images))
        #     this_picture_images = []
        #     target_img = None
        #     print(str(frame) + ' frame finished')



if __name__ == '__main__':
    main()
