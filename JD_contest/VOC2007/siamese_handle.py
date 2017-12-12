#!/usr/bin/env python


import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


test_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/test_B_resnet50.npy'
train_dir = ''

def main():
    test_data = np.load(test_dir)
    print(test_data.shape)
    for test_sqe in range(test_data.shape[0]):
        print(test_data[test_sqe])  # train_data
        test_result = []
        for pig in range(1,31):
            train_data = np.load(os.path.join(train_dir, str(pig)))
            num = 0
            for train_sqe in range(train_data.shape[0]):
                euclidean_distance = F.pairwise_distance(test_data[test_sqe], train_data[train_sqe])
                num += euclidean_distance
            average = num/train_data.shape[0]
            test_result.append(average)
        output = nn.functional.softmax(test_result)



if __name__ == '__main__':
    main()


# euclidean_distance = F.pairwise_distance(output1[:, :, 0, 0], output2[:, :, 0, 0])
