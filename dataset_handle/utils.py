#!/usr/bin/env python
#coding=utf-8

##---this is the utils file of the project---##

import os
import time

def count_file(path):
    ls = os.listdir(path)
    count = 0
    for i in ls:
        if(os.path.isfile(os.path.join(path,i))):
            count += 1
    return count

def build_dir(path):
    if not os.path.exists(path):    # build new folder
        os.makedirs(path)

def copyFiles(sourceFile,  targetFile):  # copy
    if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):  
                    open(targetFile, "wb").write(open(sourceFile, "rb").read()) 

def get_time():
    return time.asctime(time.localtime(time.time()))


