#!/usr/bin/env python
#coding=utf-8

# 用于将darknet的label转化成keras版yolo需要的label格式
# image_file_path box1 box2 ... boxN

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


sets = ['train', 'val']
dataset_dir = '/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/turtlebot'
classes = ["t", "car"]
dataset_name = 'turtlebot'



def convert_annotation(image_id, list_file):
    in_file = open('%s/Annotations/%s.xml'%(dataset_dir, image_id))
    out_file = open('%s/labels/%s.txt'%(dataset_dir, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

for image_set in sets:
    if not os.path.exists(dataset_dir):
        raise IOError('There is no dataset!!')
    # if not os.path.exists('%s/labels'%dataset_dir):
    #     os.makedirs('%s/labels'%dataset_dir)
    # output_file = os.path.join(dataset_dir, image_set+'.txt')
    # file = open(output_file, 'w')
    image_ids = open('%s/ImageSets/Main/%s.txt'%(dataset_dir, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(dataset_name, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/JPEGImages/%s.jpg'%(dataset_dir, image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')

    list_file.close()

os.system("cat %s_train.txt %s_val.txt > train.txt"%(dataset_name, dataset_name))
# os.system("cat %s_train.txt %s_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")



# import os
#
#
# darknet_sample_list_dir = '/home/zq610/WYZ/deeplearning/network/darknet/train_test_list/turtlebot/train.txt'
# output_keras_sample_list_dir = '/home/zq610/WYZ/tiny_code/dataset_handle/keras_train.txt'
#
#
# darknet_sample_list_file = open(darknet_sample_list_dir, 'r')
# darknet_sample_list = darknet_sample_list_file.readlines()
# output_keras_sample_list_flie = open(output_keras_sample_list_dir, 'w')
#
#
# for sample in darknet_sample_list:
#     output = []
#     objects = []
#
#     sample_name = sample.split('/')[-1].split('.')[0]
#     output.append(sample_name)
#     print(sample_name)
#     father_dir = os.path.abspath(os.path.dirname(sample) + '/..')
#     print(father_dir)
#     label_file = open(os.path.join(father_dir, 'labels', '{name}.txt'.format(name=sample_name)), 'r')
#     for object in label_file:
#
#     print(label_file)
#     exit()
#
#
# print(darknet_sample_list)

