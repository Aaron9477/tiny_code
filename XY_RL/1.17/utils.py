#!/usr/bin/env python

import os
import csv
import argparse
import numpy as np
import time
import datetime
import math

def get_time():
    return time.asctime(time.localtime(time.time()))

def get_string_time():
    t = datetime.datetime.now()
    time_string = "%s-%s-%s-%s:%s" % (t.year,t.month,t.day,t.hour,t.minute)
    return time_string

def build_dir(path):
    if not os.path.exists(path):    # build new folder
        os.makedirs(path)

def copyFiles(sourceFile,  targetFile):  # copy
    if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
                    open(targetFile, "wb").write(open(sourceFile, "rb").read())

def search(path=".", name="1"):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            search(item_path, name)
        elif os.path.isfile(item_path):
            if name in item:
                print(item_path)
                return search()

def square(x):
    return x*x

def error_square(input, value):
    sum = 0
    for i in range(len(input)):
        sum += square(input[i] - value)
    return math.sqrt(sum)