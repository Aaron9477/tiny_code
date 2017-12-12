#!/usr/bin/env python

import os
import utils as u

src_dir = '/home/zq610/WYZ/JD_contest/wipe_out/filling_black'
folder = ['tian0', 'tian01', 'tian02']
SUFFIX = '.jpg'

def main():
    try:
        for folder_seq in range(3):
            image_sir = os.path.join(src_dir, folder[folder_seq])
            for pig in range(1+10*folder_seq, 11+10*folder_seq):
                num = 1
                pig_folder = os.path.join(src_dir, str(pig))
                u.build_dir(pig_folder)
                for file in os.listdir(image_sir):
                    if file.startswith(str(pig) + '_'):
                        image_name = str(pig) + '_' + str(num) + SUFFIX
                        u.copyFiles(os.path.join(image_sir, file), os.path.join(pig_folder, image_name))
                        print(file + ' finished transferring')
                        num += 1
                print(str(pig) + ' finished!')
    except Exception as e:
        # os.system("shutdown -t 5")
        raise e

    # os.system("shutdown -t 5")

        



if __name__ == '__main__':
    main()