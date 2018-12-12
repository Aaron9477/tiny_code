#!/usr/bin/env python
#coding=utf-8

import os

label_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/turtlebot/VOC_type/turtlebot/labels"
num_type = 2

total = []
for i in range(num_type):
    total.append(0)
for label in os.listdir(label_dir):
    tmp = open(os.path.join(label_dir, label))
    content = tmp.readlines()
    for line in content:
        line_type = int(line.split(' ')[0])
        total[line_type] += 1

print("the proportion of type is:")
output = ''
for i in total:
    output = output + str(i) + ' : '
output = output[:-2]
print(output)
