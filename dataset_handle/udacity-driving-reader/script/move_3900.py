#!/usr/bin/env python
# coding=utf-8

import os
import re
import glob

image_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/udacity_dataset2/training/HMB_1/images"
target_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/udacity_dataset2/validation/HMB_1_501/"
images = [os.path.basename(x) for x in glob.glob(image_dir + "/*.png")]    # glob用于通配符匹配,这里就是找png文件

im_stamps = []
for im in images:
    stamp = int(re.sub(r'\.png$', '', im))
    im_stamps.append(stamp)
im_stamps.sort()
val_dataset = im_stamps[3900:]

for pic in val_dataset:
    pic_dir = os.path.join(image_dir, (str(pic)+'.png'))
    os.system('mv {source} {target}'.format(source=pic_dir, target=target_dir))
print('finished!!')
    # print(pic_dir)
    # exit()
# print(val_dataset)
# print(len(val_dataset))
# im_stamps = np.array(sorted(im_stamps))

