#!/usr/bin/env python

# 将输出转化成提交的格式

import os
import csv
import argparse
import numpy as np
import utils as u
from  sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import joblib

result_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy/result_xgb'
test_name_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy/test'

def main():
    result = np.load(os.path.join(result_dir, 'result.npy'))
    pig_list = np.load(os.path.join(test_name_dir, 'test_A_name_resnet50.npy'))
    row_result = 0  # the row of result handled now
    all_result = [] # save all the handled date
    for pig in result:
        for predict in range(30):
            tmp_array = [pig_list[row_result][0], predict+1, pig[predict]]
            tmp_array = u.str2num(tmp_array)
            all_result.append(tmp_array)
        row_result += 1
    output_dir = os.path.join(result_dir, 'tidy_out.csv')
    u.write_csv(all_result, output_dir)

    print(all_result)



if __name__ == '__main__':
    main()