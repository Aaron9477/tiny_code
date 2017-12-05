#!/usr/bin/env python

# 工具文件

import os
import csv
import argparse
import numpy as np
import time



def get_time():
    return time.asctime(time.localtime(time.time()))

def build_dir(path):
    if not os.path.exists(path):    # build new folder
        os.makedirs(path)

def str2num(input):    # str to number
    input[0] = int(input[0])
    input[1] = int(input[1])
    input[2] = float(input[2])
    return input


def write_csv(input, output_dir):  # write out
    csvFILE = open(output_dir, 'w')
    csv_writer = csv.writer(csvFILE)
    for i in range(len(input)):
        csv_writer.writerow(input[i])
    csvFILE.close()  # finish all the input, and do this code!!!!!!!!!!!!!!!!!!!!!
    print('all finished')


def process_before_write(input):
    input[2] = input[2] / 1.00001
    return input

