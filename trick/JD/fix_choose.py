#!/usr/bin/env python
import os
import random


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


def main(random_rate, src_path, save_train_path, save_val_path):
    for pig in range(1,31):
        print("the {s} pig starts".format(s=pig))   # display
        pig_path = src_path + "/" + str(pig)
        pig_train_path = os.path.join(save_train_path, str(pig))
        pig_val_path = os.path.join(save_val_path, str(pig))

        build_dir(pig_train_path)
        build_dir(pig_val_path)

        num_image = count_file(pig_path)    
        num_train = int(num_image * random_rate)    # conut num of train
        #print(num_image, num_train)
        train_choose = range(1, num_train+1)  # get train fixly
        val_choose = range(num_train, num_image+1) # get the ramain
        #print(train_choose)
        #print(val_choose)

        num = 1
        for train_image in train_choose:
            src_image = pig_path + "/" + str(train_image) + ".jpg"  # source file
            save_image = pig_train_path + "/" + str(num) + ".jpg"    # target file
            copyFiles(src_image, save_image)
            num += 1
        num = 1
        for val_image in val_choose:
            src_image = pig_path + "/" + str(val_image) + ".jpg"  # source file
            save_image = pig_val_path + "/" + str(num) + ".jpg"    # target file
            copyFiles(src_image, save_image)
            num += 1


if __name__ == '__main__':
    # where to change 
    div_rate = 0.7
    root_path = '/home/zq610/WYZ/JD_contest'
    interval = 10


    src_path = root_path + '/refine_frame/interval_' + str(interval)
    save_train_path = root_path + '/train_set/train'
    save_val_path = root_path + '/train_set/val'
    build_dir(save_train_path)
    build_dir(save_val_path)    
    main(div_rate, src_path, save_train_path, save_val_path)
    print("all finished")