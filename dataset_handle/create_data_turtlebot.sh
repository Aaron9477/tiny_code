#!/bin/bash

# 该脚本修改自mobilenet-caffe
# 用于从VOC格式的数据集制作lmdb文件

# cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
# root_dir=$cur_dir/../..
root_dir=/home/zq610/WYZ/deeplearning/network/ssd-caffe/caffe

cd $root_dir

redo=1
# data_root_dir="$HOME/WYZ/deeplearning/network/ssd-caffe/caffe/data/VOCdevkit/"
data_root_dir=/home/zq610/WYZ/deeplearning/network/ssd-caffe/caffe/data/VOC_type
config_dir="$data_root_dir/turtlebot_test"

dataset_name="turtlebot"
mapfile="$config_dir/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi

python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim \
--max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label \
$extra_cmd $data_root_dir $config_dir/test.txt \
$data_root_dir/$dataset_name/$db/$dataset_name"_"test"_"$db examples/$dataset_name

