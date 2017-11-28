#!/bin/bash  
  
for((i=1;i<=5;i++));  
do   
python finetune_classifier_cross.py /home/zq610/WYZ/JD_contest/wipe_out/good/ --pretrained -a resnet50 -s $i
done  
shutdown -h 5

