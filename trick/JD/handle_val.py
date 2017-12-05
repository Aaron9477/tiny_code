#!/usr/bin/env python

import os
import csv
import argparse
import matplotlib.pyplot as plt
from pylab import *                                 

parser = argarse.ArgumentParser(description = 'visualize the val')
parser.add_argument('data', metavar='root_path', default='/home/zq610/WYZ/JD_contest/process/17-11-26-2/log_resnet50/val/1/val.csv',help='path to the val')
parser.add_argument('-m', '--mode', default=1, type=int, metavar='draw mode',
                    help='1:draw one, 2:draw cross')



def show_val_change(csv_path):
    csv_reader = csv.reader(open(csv_path))
    val = []
    for row in csv_reader:
        out_put = row[0].split(' ')
        val.append(out_put[-1])
    print(val)
    val = map(float, val)
    plt.plot(range(1, len(val)+1), val)
    plt.xlabel('epoch')
    plt.ylabel("val_loss")
    plt.title("val_loss change")
    plt.show()
    return val


def main():
    args = parser.parse_arg()
    if(args.mode == 1):
        show_val_change(args.data)
    elif(args.mode == 2):
        cross_val = []
        for epoch in range(1,6):
            val_dir = os.path.join(args.data, str(epoch))
            cross_val.append(show_val_change(val_dir))
    else:
        print('wrong input')
        exit()



if __name__ == '__main__':
    main()