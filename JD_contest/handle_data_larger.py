#!/usr/bin/env python

import os
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'handle the result')
parser.add_argument('data', metavar='root_path', default='/home/zq610/WYZ/JD_contest/input.csv',help='path to the val')
output_dir = '/home/zq610/WYZ/JD_contest/output.csv'

def str2num(input):    # str to number
    input[0] = int(input[0])
    input[1] = int(input[1])
    input[2] = float(input[2])
    return input

def judge_num(remain_list, input_list, threshold):
    list_over_threshold = []
    for i in remain_list:
        if input_list[i][2] > threshold:
            # remain_list.remove(i)
            list_over_threshold.append(i)
    return list_over_threshold
            

def process_data(input):   # process data
    # print(input)
    remain_list = range(30) # all the 30 list, waiting removing
    list_over_09 = judge_num(remain_list, input, 0.9)
    if len(list_over_09) > 0:   #0.9
        remain_possible = 1
        for a in list_over_09:
            input[a][2] = 0.8
            remain_list.remove(a)
            remain_possible -= 0.8
        for b in remain_list:
            input[b][2] = round(remain_possible/len(remain_list), 6)
        return input  

    list_over_08 = judge_num(remain_list, input, 0.8)
    if len(list_over_08) > 0:   #0.8
        remain_possible = 1
        for a in list_over_08:
            input[a][2] = 0.7
            remain_list.remove(a)
            remain_possible -= 0.7
        list_over_01 = judge_num(remain_list, input, 0.1)
        if len(list_over_01) > 0:  #0.8+0.1+0.1
            for b in list_over_01:
                input[b][2] = 0.1
                remain_list.remove(b)
                remain_possible -= 0.1
            for c in remain_list:
                input[c][2] = round(remain_possible/len(remain_list), 6)
        else:
            for b in remain_list:
                input[b][2] = round(remain_possible/len(remain_list), 6)
        return input


    list_over_07 = judge_num(remain_list, input, 0.7)
    if len(list_over_07) > 0:   #0.7
        remain_possible = 1
        for a in list_over_07:
            input[a][2] = 0.55
            remain_list.remove(a)
            remain_possible -= 0.55
        list_over_02 = judge_num(remain_list, input, 0.2)
        if len(list_over_02) > 0:   #0.7+0.2
            input[list_over_02[0]][2] = 0.15
            remain_list.remove(list_over_02[0])
            remain_possible -= 0.15
            for a in remain_list:
                input[a][2] = round(remain_possible/len(remain_list), 6)
            return input
        list_over_01 = judge_num(remain_list, input, 0.1)
        if len(list_over_01) == 2:  #0.7+0.1+0.1
            for a in list_over_01:
                input[a][2] = 0.1
                remain_list.remove(a)
                remain_possible -= 0.1
            for b in remain_list:
                input[b][2] = round(remain_possible/len(remain_list), 6)
        elif len(list_over_01) == 1:    #0.7+0.1
            for a in list_over_01:
                input[a][2] = 0.1
                remain_list.remove(a)
                remain_possible -= 0.1
            for b in remain_list:
                input[b][2] = round(remain_possible/len(remain_list), 6)
        else:
            for b in remain_list:
                input[b][2] = round(remain_possible/len(remain_list), 6)                
        return input


    list_over_06 = judge_num(remain_list, input, 0.6)
    if len(list_over_06) > 0:   #0.7
        remain_possible = 1
        for a in list_over_06:
            input[a][2] = 0.4
            remain_list.remove(a)
            remain_possible -= 0.4

        list_over_03 = judge_num(remain_list, input, 0.3)
        if len(list_over_03) > 0:   #0.6+0.3
            input[list_over_03[0]][2] = 0.2
            remain_list.remove(list_over_03[0])
            remain_possible -= 0.2
            for a in remain_list:
                input[a][2] = round(remain_possible/len(remain_list), 6)
            return input

        list_over_02 = judge_num(remain_list, input, 0.2)
        if len(list_over_02) > 0:   #0.6+0.2
            input[list_over_02[0]][2] = 0.15
            remain_list.remove(list_over_02[0])
            remain_possible -= 0.15
            list_over_01 = judge_num(remain_list, input, 0.1)
            if len(list_over_01) > 0:   #0.6+0.2+0.1
                input[list_over_01[0]][2] = 0.1
                remain_list.remove(list_over_01[0])
                remain_possible -= 0.1
                for a in remain_list:
                    input[a][2] = round(remain_possible/len(remain_list), 6)
            else:
                for b in remain_list:
                    input[b][2] = round(remain_possible/len(remain_list), 6)
            return input

        list_over_01 = judge_num(remain_list, input, 0.1)
        if len(list_over_01) > 0:   #0.6+0.2+0.1
            for a in list_over_01:
                input[a][2] = 0.1
                remain_list.remove(a)
                remain_possible -= 0.1
            for b in remain_list:
                input[b][2] = round(remain_possible/len(remain_list), 6)
        else:
            for b in remain_list:
                input[b][2] = round(remain_possible/len(remain_list), 6)
        return input


    list_over_05 = judge_num(remain_list, input, 0.5)
    if len(list_over_05) > 0:   #0.6+0.2
        remain_possible = 1
        for a in list_over_05:
            input[a][2] = 0.25
            remain_list.remove(a)
            remain_possible -= 0.25
        list_over_03 = judge_num(remain_list, input, 0.3)
        if len(list_over_03) > 0:   #0.6+0.2+0.1
            for b in list_over_03:
                input[b][2] = 0.15
                remain_list.remove(b)
                remain_possible -= 0.5
        list_over_02 = judge_num(remain_list, input, 0.2)
        if len(list_over_02) > 0:   #0.6+0.2+0.1
            for b in list_over_02:
                input[b][2] = 0.1
                remain_list.remove(b)
                remain_possible -= 0.1
        for c in remain_list:
            input[c][2] = round(remain_possible/len(remain_list), 6)
        return input


    list_over_04 = judge_num(remain_list, input, 0.4)
    if len(list_over_04) > 0:   #0.6+0.2
        remain_possible = 1
        for a in list_over_04:
            input[a][2] = 0.2
            remain_list.remove(a)
            remain_possible -= 0.2
        list_over_02 = judge_num(remain_list, input, 0.2)
        if len(list_over_02) > 0:   #0.6+0.2+0.1
            for b in list_over_02:
                input[b][2] = 0.1
                remain_list.remove(b)
                remain_possible -= 0.1
        for c in remain_list:
            input[c][2] = round(remain_possible/len(remain_list), 6)
        return input        


    list_over_03 = judge_num(remain_list, input, 0.3)
    if len(list_over_03) > 0:   #0.6+0.2
        remain_possible = 1
        for a in list_over_03:
            input[a][2] = 0.2
            remain_list.remove(a)
            remain_possible -= 0.2
        for c in remain_list:
            input[c][2] = round(remain_possible/len(remain_list), 6)
        return input        

    for i in remain_list:
        input[i][2] = round(1.0/len(remain_list), 6)
    return input


def write_csv(input, csvFILE):  # write out
    
    csv_writer = csv.writer(csvFILE)
    for i in range(len(input)):
        csv_writer.writerow(input[i])

def process_before_write(input):
    input[2] = input[2]/1.0001
    return input



def main():
    args = parser.parse_args()
    csv_reader = csv.reader(open(args.data))
    csvFILE = open(output_dir, 'w')
    while(True):
        try:
            num = 0
            thirty_pigs = []
            while (num < 30):
                tmp = str2num(csv_reader.next())    # read and transfer to number
                thirty_pigs.append(tmp)
                num += 1
            processed_data = process_data(thirty_pigs)
            print(processed_data)   
            processed_data = map(process_before_write, processed_data)
            write_csv(processed_data, csvFILE)

            print(processed_data)
                
        except Exception as e:
            csvFILE.close() # finish all the input, and do this code!!!!!!!!!!!!!!!!!!!!!
            print('all finished')
            raise e

    # for 30 in csv_reader:
    # print(csv_reader.next())
    # num = 0
    # thirty_pigs = []
    # while num < 30:

    

if __name__ == '__main__':
    main()


def get_img_size(input_dir, image_name):
    tmp_img_dir = os.path.join(input_dir, image_name)    #get img's dir
    tmp_img_info = cv2.imread(tmp_img_dir)
    tmp_img_shape = tmp_img_info.shape
    tmp_img_size = tmp_img_shape[0] * tmp_img_shape[1]
    return tmp_img_size