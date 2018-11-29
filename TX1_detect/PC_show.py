#!/usr/bin/env python
#  -*- coding: utf-8

import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
import cv2
from fasterrcnn2.msg import output


roi_region = roi_origin = [0,0,0,0]
global roi_region
faster_FLAG = False
global faster_FLAG
def getCVimage(raw_image):
    try:
        global roi_region
        TX1_frame = CvBridge().imgmsg_to_cv2(raw_image, "bgr8")
        rospy.loginfo("transfer ros_image to cv_image")
        if roi_region != roi_origin:
            cv2.rectangle(TX1_frame, (roi_region[0], roi_region[1]), (roi_region[2],  roi_region[3]), (0, 255, 0), 5)
        cv2.namedWindow("TX1_view")
        cv2.imshow("TX1_view", TX1_frame)
        cv2.waitKey(1)
        # 下面的不好用,不知道为啥
        # if cv2.waitKey(1) == 'q':  # 要显示必须加waitkey
        #     print('shutdown!!')
        #     exit()
    except CvBridgeError as e:
        print (e)

def show_roi(roi):
    global roi_region
    roi_region = [roi.x_offset, roi.y_offset, roi.x_offset+roi.width, roi.y_offset+roi.height]

def faster_get_roi():
    global faster_FLAG
    faster_FLAG = True

# def shutdown():
#     print("shutdown!!")


rospy.init_node("PC_shower")    # 没有这句话就不能继续!!!必须要初始化一个节点

rospy.loginfo("Waiting for image topic...")
rospy.wait_for_message('/camera/rgb/image_raw', Image)

# 这里不需要定义rawimage_sub,如果需要后面不再订阅这个话题,需要定义.再使用取消
raw_image = rospy.Subscriber("/camera/rgb/image_raw", Image, getCVimage)
raw_roi = rospy.Subscriber('tld_roi', RegionOfInterest, show_roi)

rospy.loginfo("Waiting for fasterRCNN detect the goal...")
faster_roi = rospy.Subscriber('fasterrcnn', output, faster_get_roi)

while not faster_FLAG:


# rospy.on_shutdown(shutdown)

rospy.spin()

# while not rospy.is_shutdown():
#     rospy.spinOnce()