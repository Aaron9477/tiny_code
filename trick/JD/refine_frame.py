#!/usr/bin/env python
import os

def copyFiles(sourceFile,  targetFile):  # copy
    if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):  
                    open(targetFile, "wb").write(open(sourceFile, "rb").read()) 

def main(src_path, save_path, interval, last_picture):
    new_path = save_path + '/interval_' + str(interval)
    if not os.path.exists(new_path):    # build new folder
        os.makedirs(new_path)

    for pig in range(1,31): # 1 to 30 pig
        print("the {s} pig starts".format(s=pig))   # display
        save_path_tmp = new_path + "/" + str(pig)
        if not os.path.exists(save_path_tmp):
            os.mkdir(save_path_tmp)
        num = 1 #sequence
        for i in range(1, last_picture, interval):
            src_image = src_path + "/" + str(pig) + "/" + str(pig) + "_" + str(i) + ".jpg"  # source file
            save_image = save_path_tmp + "/" + str(num) + ".jpg"    # target file
            copyFiles(src_image, save_image)
            num += 1   

if __name__ == '__main__':
    # change follows
    src_path = '/home/zq610/WYZ/JD_contest/raw_frame'
    save_path = '/home/zq610/WYZ/JD_contest/refine_frame'
    interval = 5  #serial interval
    last_picture = 2951
    main(src_path, save_path, interval, last_picture)
    print("all finished")