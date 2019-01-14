
制作数据集步骤说明:
1.使用interval_choose_and_crop对视频进行跟踪,得到数据集
2.使用labelImg.py进行精修,注意使用python3运行

yolo分支
3-yolo.使用write_imagesets.py制作list文件
4-yolo.使用yolo自带的darknet/scripts/voc_label_wyz.py文件进行yolo的label制作
5-yolo.将得到的在darknet目录下的train.ext替换,勿忘!!!!
6-yolo.之后使用yolo_kmeans.py文件,使用kmeans对bbox进行聚类
7-yolo.使用聚类结果对cfg文件进行修改
8-yolo.可通过cal_diffent_type.py进行类别比例的计算
至此完成yolo数据集的制作
训练需要注意的:
改变scale在detector.c中

caffe lmdb分支
按照https://github.com/weiliu89/caffe/tree/ssd
https://github.com/chuanqi305/MobileNet-SSD
这两个人网站上的指示进行操作
3-caffe.使用create_list_turtlebot.sh和create_data_turtlebot.sh制作出list和lmdb文件
