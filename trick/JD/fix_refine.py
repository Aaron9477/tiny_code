#!/usr/bin/env python
import os
import argparse


parser = argparse.ArgumentParser(description='refine from raw pig images')
parser.add_argument('data', metavar='root_path',default='/home/zq610/WYZ/JD_contest',
                    help='path to dataset')
parser.add_argument('-r', '--rate', default=0.6, type=float, metavar='div_rate',
                    help='The rate of train and val')
parser.add_argument('-i', '--interval', default=10, type=int, metavar='choose interval',
                    help='interval of frame')


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


def main(div_rate, src_path, save_train_path, save_val_path):
    for pig in range(1,31):
        print("the {s} pig starts".format(s=pig))   # display
        pig_src_path = os.path.join(src_path, str(pig))
        pig_train_path = os.path.join(save_train_path, str(pig))
        pig_val_path = os.path.join(save_val_path, str(pig))

        build_dir(pig_train_path)
        build_dir(pig_val_path)

        num_image = count_file(pig_src_path)    
        num_train = int(num_image * div_rate)    # conut num of train
        #print(num_image, num_train)
        train_choose = range(1, num_train+1)  # get train fixly
        val_choose = range(num_train, num_image+1) # get the ramain
        #print(train_choose)
        #print(val_choose)

        num = 1
        for train_image in range(1, num_train-1, interval):
            image_name = str(pig) + "_" + str(train_image) + ".jpg"
            src_image = os.path.join(pig_src_path, image_name) # source file
            save_image = os.path.join(pig_train_path, str(num)+".jpg")    # target file
            copyFiles(src_image, save_image)
            num += 1
        num = 1
        for val_image in range(num_train+interval, num_image-1, interval):
            image_name = str(pig) + "_" + str(val_image) + ".jpg"
            src_image = os.path.join(pig_src_path, image_name)  # source file
            save_image = os.path.join(pig_val_path, str(num)+".jpg")   # target file
            copyFiles(src_image, save_image)
            num += 1


if __name__ == '__main__':
    args = parser.parse_args()
    # where to change 
    div_rate = args.rate
    root_path = args.data
    interval = args.interval


    src_path = root_path + '/raw_frame'
    save_train_path = root_path + '/train_set/train'
    save_val_path = root_path + '/train_set/val'
    build_dir(save_train_path)
    build_dir(save_val_path)    
    main(div_rate, src_path, save_train_path, save_val_path)
    print("all finished")