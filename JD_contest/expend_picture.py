#!/usr/bin/env python

from matplotlib import pyplot as plt
import cv2
import os
import utils as u

src_dir = '/home/zq610/WYZ/JD_contest/test/test_B'
output_dir = '/home/zq610/WYZ/JD_contest/test/filling_B'
BLACK = [0,0,0]

def main():
    u.build_dir(output_dir)
    for image_name in os.listdir(src_dir):
        img = cv2.imread(os.path.join(src_dir, image_name))
        height, width = img.shape[0], img.shape[1]

        if height > width:
            expend_scale = (height - width)/2
            # expend_image = cv2.copyMakeBorder(img, 0,0,expend_scale,expend_scale,cv2.BORDER_CONSTANT, value=BLACK)
            expend_image = cv2.copyMakeBorder(img, 0, 0, expend_scale, expend_scale, cv2.BORDER_REFLECT)
        elif height < width:
            expend_scale = (width - height)/2
            # expend_image = cv2.copyMakeBorder(img, expend_scale, expend_scale, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            expend_image = cv2.copyMakeBorder(img, expend_scale, expend_scale, 0, 0, cv2.BORDER_REFLECT)
        else:
            expend_image = img

        cv2.imwrite(os.path.join(output_dir, image_name), expend_image)
        print(expend_image.shape)
        print(image_name + ' has been handled!')
    print('all finished')



if __name__ == '__main__':
    main()

    # img = cv2.imread('/home/zq610/WYZ/JD_contest/test/test_A/6.JPG')
    # img1 = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    # replace = cv2.copyMakeBorder(img, 200,200,200,200,cv2.BORDER_CONSTANT, value=BLACK)
    #
    # plt.imshow(replace)
    # plt.show()