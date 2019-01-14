import os

target_dir = "/media/zq610/2C9BDE0469DC4DFC/ubuntu/dl_dataset/Dornet/udacity_dataset2"


def change_file_name(dir):
    for fil in os.listdir(dir):
        tmp_dir = os.path.join(dir, fil)
        if os.path.isdir(tmp_dir):
            if fil == 'left' or fil == 'right':
                os.system("rm -r {a}".format(a=tmp_dir))
                # print(tmp_dir)
                continue
            change_file_name(tmp_dir)

if __name__ == "__main__":
    change_file_name(target_dir)

    
      