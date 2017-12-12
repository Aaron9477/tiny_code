#!/usr/bin/env python

# 用机器学习算法进行训练，存储模型并得到结果，需要修改use_model，以及选择使用64 65行的算法

import os
import csv
import argparse
import numpy as np
import utils as u
from  sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import joblib


use_model = 'xgb'

train_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy'
test_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy/test/test_A_resnet50.npy'
output_dir = '/home/zq610/WYZ/JD_contest/wipe_out/wipe_out_small/npy/result_' + use_model


def get_feature():
    if os.path.exists(train_dir):
        feature = np.load(os.path.join(train_dir, 'train_feature.npy'))
        tar = np.load(os.path.join(train_dir, 'train_target.npy'))
    else:
        print('wrong directory of feature')
        exit()
    return feature, tar

def get_test():
    if os.path.exists(test_dir):
        test = np.load(test_dir)
    else:
        print('wrong directory of feature')
        exit()
    return test   


def get_model_rf(feature, tar):
    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=7)
    clf.fit(feature, tar)
    return clf

def get_model_xgb(feature, tar):
    clf = XGBClassifier(learning_rate=0.1,
                        n_estimators=50,
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='multi:softprob',
                        nthread=7,
                        scale_pos_weight=1,
                        seed=0)
    clf.fit(feature, tar)
    return clf


def main():
    feature, tar = get_feature()
    print(u.get_time(), ' train starts!')
    # clf = get_model_rf(feature, tar)
    clf = get_model_xgb(feature, tar)
    print(u.get_time(), ' train done!')
    u.build_dir(output_dir)
    model_save_dir = os.path.join(output_dir, use_model + '_model.m')
    joblib.dump(clf, model_save_dir)
    test = get_test()
    print(u.get_time(), ' test starts!')
    result = clf.predict_proba(test)
    print(u.get_time(), ' test done!')
    np.savetxt(os.path.join(output_dir, 'result.csv'), result, fmt='%.6f', delimiter=',')
    np.save(os.path.join(output_dir, 'result.npy'), result)
    print('all finished')




if __name__ == '__main__':
    # a = get_test()
    # print(len(a))
    # exit()
    main()