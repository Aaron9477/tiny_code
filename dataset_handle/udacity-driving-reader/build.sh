#!/bin/bash
IMAGE_TAG="udacity-reader"

# 如果后面不跟参数的话,这个while没有作用
while getopts ":t:" opt; do # 类似于switch,需要以 -t xxx 来传输,如果是t就执行下面t这个选项,$OPTARG就是-t后面的xxx,\?)是指的其它输入
  case $opt in
    t) IMAGE_TAG=$OPTARG ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
shift $(expr $OPTIND - 1)

docker build -t $IMAGE_TAG .
