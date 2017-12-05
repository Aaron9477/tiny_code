#!/usr/bin/env python

# 将30头猪的特征信息转化成两个文件，训练集train_feature.npy和对应的结果'train_target.npy'

import os
import numpy as np

feature_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy/train'
output_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy'
PREFIX = '_resnet50'
SUFFIX = '.npy'

def main():
    for seq in range(1,31):
        npy_name = str(seq) + PREFIX + SUFFIX   # file name
        tmp_array = np.load(os.path.join(feature_dir, npy_name))    # get the array
        tmp_target = np.repeat(seq, tmp_array.shape[0])# get the result of this array
        if seq == 1:
            all_feature = tmp_array
            all_target = tmp_target
        else:
            all_feature = np.row_stack((all_feature, tmp_array))    # one by one with same dimension
            all_target = np.concatenate((all_target, tmp_target))   # end to end

    np.save(os.path.join(output_dir, 'train_feature.npy'), all_feature)
    np.save(os.path.join(output_dir, 'train_target.npy'), all_target)

    print(all_feature, all_target)
    print(all_target.shape[0])

    # print(all_feature[0])
    # print(tmp_array[0])





if __name__ == '__main__':
    main()