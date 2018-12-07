#!/bin/bash

root_dir="$HOME/WYZ/deeplearning/network/ssd-caffe/caffe/data/VOC_type"
name=turtlebot_test
sub_dir=ImageSets/Main
# ${BASH_SOURCE[0]}表示bash命令的第一个参数,"dirname"表示提取参数里的目录
# 所以这里是这个shell文件坐在的路径/home/zq610/WYZ/tiny_code/dataset_handle
# bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    
bash_dir=$root_dir    
dst_file=$bash_dir/$name/test.txt

if [ -f $dst_file ]
then 
    rm -f $dst_file
fi

echo "Create list for trainval..."
dataset_file=$root_dir/$name/$sub_dir/test.txt

img_file=$bash_dir/$name/"trainval_img.txt"
cp $dataset_file $img_file
# s/（1）/（2）/g”表示查找（1），并用（2）进行替换
# ^匹配一行的开头,$匹配一行的结束
sed -i "s/^/$name\/JPEGImages\//g" $img_file
sed -i "s/$/.jpg/g" $img_file

label_file=$bash_dir/$name/"trainval_label.txt"
cp $dataset_file $label_file
sed -i "s/^/$name\/Annotations\//g" $label_file
sed -i "s/$/.xml/g" $label_file
# paste -d后面直接加的是,以什么来分隔
paste -d' ' $img_file $label_file >> $dst_file

rm -f $label_file
rm -f $img_file

# handle test dataset!!!!!!!!!!!!!!!!!!!!!!!!!!!!
/home/zq610/WYZ/deeplearning/network/ssd-caffe/caffe/build/tools/get_image_size $root_dir $dst_file $bash_dir/"test_name_size.txt"

# handle trainval dataset!!!!!!!!!!!!!!!!!!!!!!!!
# Shuffle trainval file.
# rand_file=$dst_file.random
# cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
# mv $rand_file $dst_file

