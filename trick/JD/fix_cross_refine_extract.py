#!/usr/bin/env python
import os
import argparse

##----From raw images' folder, it will refine image whit fixed interval, ant in these chose images it will divide it with fixed rate---##
##---the different of it with fix_refine.py is that it don't have fixed name of image, so it can work every where---##


parser = argparse.ArgumentParser(description='refine from raw pig images')
parser.add_argument('data', metavar='root_path',default='/home/zq610/WYZ/JD_contest/roi',
                    help='path to dataset')
parser.add_argument('-r', '--rate', default=0.8, type=float, metavar='div_rate',
                    help='The rate of train and val')
parser.add_argument('-i', '--interval', default=1, type=int, metavar='choose interval',
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

def list_of_groups(init_list, childern_list_len):
    list_of_groups = zip(*(iter(init_list),) *childern_list_len)    #divide into some group
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % childern_list_len
    end_list.append(init_list[-count:]) if count !=0 else end_list
    return end_list


def main(div_rate, src_path, save_train_path, save_val_path, interval, i):
    for pig in range(1,31):
        print("the {s} pig starts".format(s=pig))   # display
        pig_src_path = os.path.join(src_path, str(pig))
        # print(os.listdir(pig_src_path))
        pig_train_path = os.path.join(save_train_path, str(pig))
        pig_val_path = os.path.join(save_val_path, str(pig))

        build_dir(pig_train_path)
        build_dir(pig_val_path)

        num_image = count_file(pig_src_path)    
        images_all=os.listdir(pig_src_path) # get all the files list
        num_val = int(num_image / interval * (1 - div_rate)) + 1

        images_group = list_of_groups(images_all,num_val)
        # print(images_group[4])
        #num_train = int(num_image * random_rate)    # conut num of train
        #print(num_image, num_train)
        # train_choose = range(1, num_train+1)  # get train fixly
        # val_choose = range(num_train+1, num_image+1) # get the ramain
        if i==1:
            train_choose = images_group[1]+images_group[2]+images_group[3]+images_group[4]  # get train fixly
            val_choose = images_group[0] # get the ramain
        elif i==2:
            train_choose = images_group[0]+images_group[2]+images_group[3]+images_group[4]  # get train fixly
            val_choose = images_group[1] # get the ramain
        elif i==3:
            train_choose = images_group[0]+images_group[1]+images_group[3]+images_group[4]  # get train fixly
            val_choose = images_group[2] # get the ramain
        elif i==4:
            train_choose = images_group[0]+images_group[1]+images_group[2]+images_group[4]  # get train fixly
            val_choose = images_group[3] # get the ramain
        else:
            train_choose = images_group[0]+images_group[1]+images_group[2]+images_group[3]  # get train fixly
            val_choose = images_group[4] # get the ramain         
        #print(train_choose)
        #print(val_choose)

        num = 1
        for train_image in train_choose:
            src_image = os.path.join(pig_src_path, train_image)
            # src_image = pig_src_path + "/" + str(pig) + "_" + str(train_image) + ".jpg"  # source file
            save_image = pig_train_path + "/" + str(num) + ".jpg"    # target file
            copyFiles(src_image, save_image)
            num += 1
        num = 1
        for val_image in val_choose:
            src_image = os.path.join(pig_src_path, val_image)
            # src_image = pig_src_path + "/" + str(pig) + "_" + str(val_image) + ".jpg"  # source file
            save_image = pig_val_path + "/" + str(num) + ".jpg"    # target file
            copyFiles(src_image, save_image)
            num += 1


if __name__ == '__main__':
    args = parser.parse_args()
    # where to change 
    div_rate = args.rate
    root_path = args.data
    interval = args.interval


    # src_path = root_path + '/raw_frame'
    src_path = root_path
    for i in range(1,6):
        save_train_path = root_path + '/train_set_' + str(i) + '/train'
        save_val_path = root_path + '/train_set_' + str(i) + '/val'
        build_dir(save_train_path)
        build_dir(save_val_path)    
        main(div_rate, src_path, save_train_path, save_val_path, interval, i)
    print("all finished")