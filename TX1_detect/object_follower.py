#!/usr/bin/python
# -*- coding: utf-8 

"""
    object_follower.py - Version 1.1 2013-12-20
    
    Follow a target published on the /roi topic using depth from the depth image.
    
    Created for the Pi Robot Project: http://www.pirobot.org
    Copyright (c) 2013 Patrick Goebel.  All rights reserved.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details at:
    
    http://www.gnu.org/licenses/gpl.html
"""

import rospy
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from geometry_msgs.msg import Twist
from math import copysign
#from cv2 import cv as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import thread

class ObjectFollower():
    def __init__(self):
        rospy.init_node("object_follower")
                        
        # Set the shutdown function (stop the robot)
        rospy.on_shutdown(self.shutdown)
        
        # How often should we update the robot's motion?
        self.rate = rospy.get_param("~rate", 10)
        
        r = rospy.Rate(self.rate)
        
        # Scale the ROI by this factor to avoid background distance values around the edges
        self.scale_roi = rospy.get_param("~scale_roi", 0.8)
        
        # The max linear speed in meters per second
        self.max_linear_speed = rospy.get_param("~max_linear_speed", 0.3)
        
        # The minimum linear speed in meters per second
        self.min_linear_speed = rospy.get_param("~min_linear_speed", 0.02)
        
        # The maximum rotation speed in radians per second
        self.max_rotation_speed = rospy.get_param("~max_rotation_speed", 0.4)
        
        # The minimum rotation speed in radians per second
        self.min_rotation_speed = rospy.get_param("~min_rotation_speed", 0.2)
        
        # The x threshold (% of image width) indicates how far off-center
        # the ROI needs to be in the x-direction before we react
        self.x_threshold = rospy.get_param("~x_threshold", 0.1)
        
        # How far away from the goal distance (in meters) before the robot reacts
        self.z_threshold = rospy.get_param("~z_threshold", 0.1)
        
        # The maximum distance a target can be from the robot for us to track
        self.max_z = rospy.get_param("~max_z", 2)

        # The minimum distance to respond to
        self.min_z = rospy.get_param("~min_z", 0.5)
        
        # The goal distance (in meters) to keep between the robot and the person
        self.goal_z = rospy.get_param("~goal_z", 0.8)

        # How much do we weight the goal distance (z) when making a movement
        self.z_scale = rospy.get_param("~z_scale", 0.3)

        # How much do we weight (left/right) of the person when making a movement        
        self.x_scale = rospy.get_param("~x_scale", 1.0)
        self.gain = rospy.get_param("~gain", 2.0)
        
        # Slow down factor when stopping
        self.slow_down_factor = rospy.get_param("~slow_down_factor", 0.8)
        
        # Initialize the global ROI
        self.roi = RegionOfInterest()

        # Publisher to control the robot's movement
        self.cmd_vel_pub = rospy.Publisher('cmd_vel_mux/input/teleop', Twist)
        
        # Intialize the movement command
        self.move_cmd = Twist()
        
        # Get a lock for updating the self.move_cmd values
        self.lock = thread.allocate_lock()
        
        # We will get the image width and height from the camera_info topic
        self.image_width = 640
        self.image_height = 480
        
        # We need cv_bridge to convert the ROS depth image to an OpenCV array
        self.cv_bridge = CvBridge()
        self.depth_array = None
        
        # Set flag to indicate when the ROI stops updating
        self.target_visible = False
        
        # Wait for the camera_info topic to become available
        rospy.loginfo("Waiting for camera_info topic...")
        
        rospy.wait_for_message('/camera/rgb/camera_info', CameraInfo)
        
        # Subscribe to the camera_info topic to get the image width and height
        rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self.get_camera_info)

        # Wait until we actually have the camera data
        while self.image_width == 0 or self.image_height == 0:
            rospy.sleep(1)
            
        # Wait for the depth image to become available
        rospy.loginfo("Waiting for depth_image topic...")
        
        rospy.wait_for_message('camera/depth/image_raw', Image)
                    
        # Subscribe to the depth image
        # 接收到后，将原始图片转化成opencv可以识别的图片（深度图片），直接订阅原始图像
        self.depth_subscriber = rospy.Subscriber("camera/depth/image_raw", Image, self.convert_depth_image, queue_size=1)
        
        # Subscribe to the ROI topic and set the callback to update the robot's motion
        # 可以接收tld发送的目标框位置
        rospy.Subscriber('tld_roi', RegionOfInterest, self.set_cmd_vel)
        
        # Wait until we have an ROI to follow
        rospy.loginfo("Waiting for an ROI to track...")
        
        rospy.wait_for_message('tld_roi', RegionOfInterest)
        
        rospy.loginfo("ROI messages detected. Starting follower...")
        
        # Begin the tracking loop
        while not rospy.is_shutdown():
            # Acquire a lock while we're setting the robot speeds
            self.lock.acquire() # 互斥锁上锁
            
            try:
                # If the target is not visible, stop the robot
                # if not self.target_visible:
                #     self.move_cmd = Twist() 
                # else:
                #     # Reset the flag to False by default
                #     self.target_visible = False
                    
                # Send the Twist command to the robot
                self.cmd_vel_pub.publish(self.move_cmd) # 发布
                
            finally:
                # Release the lock
                self.lock.release() # 解锁
                
            # Sleep for 1/self.rate seconds
            r.sleep()

    # 下面这个函数除了要注意msg的格式之外，其它用现成的就行
    def set_cmd_vel(self, msg):
        # Acquire a lock while we're setting the robot speeds
        print "I got message from tld, x_offset:{a}, y_offset:{b}, height:{c}, width:{d}".format(a=msg.x_offset, b=msg.y_offset, c=msg.height, d=msg.width)

        self.lock.acquire()
        # 因为是小车，所以只用x坐标即可
        try:
            # If the ROI has a width or height of 0, we have lost the target
            if msg.width == 0 or msg.height == 0:
                self.target_visible = False
                return
            
            # If the ROI stops updating this next statement will not happen
            self.target_visible = True
            self.roi = msg
            # Now compute the depth component
            
            # Initialize a few depth variables
            n_z = sum_z = mean_z = 0
             
            # Shrink the ROI slightly to avoid the target boundaries
            # 减小目标区域，以免选中目标的边缘
            scaled_width = int(self.roi.width * self.scale_roi)
            scaled_height = int(self.roi.height * self.scale_roi)
            
            # Get the min/max x and y values from the scaled ROI
            # 得到缩小的框
            min_x = int(self.roi.x_offset + self.roi.width * (1.0 - self.scale_roi) / 2.0)
            max_x = min_x + scaled_width
            min_y = int(self.roi.y_offset + self.roi.height * (1.0 - self.scale_roi) / 2.0)
            max_y = min_y + scaled_height

            depth_set = []
            # Get the average depth value over the ROI
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    try:
                        # Get a depth value in millimeters
                        z = self.depth_array[y, x]
                            
                        # Convert to meters
                        z /= 1000.0
                        
                        
                    except:
                        # It seems to work best if we convert exceptions to 0
                        z = 0
                        
                    # Check for values outside max range
                    if z > self.max_z:
                        continue
                    
                    # Increment the sum and count
                    depth_set.append(z)
                    sum_z = sum_z + z
                    n_z += 1 

            # Compute the displacement of the ROI from the center of the image
            target_offset_x = msg.x_offset + msg.width / 2 - self.image_width / 2  # 目标x方向中心，相对于图像中心的位置

            try:
                percent_offset_x = float(target_offset_x) / (float(self.image_width) / 2.0) # x方向误差量相对于图像一半宽度
            except:
                percent_offset_x = 0
    
            # Rotate the robot only if the displacement of the target exceeds the threshold
            if abs(percent_offset_x) > self.x_threshold:    # 如果大于阈值
                # Set the rotation speed proportional to the displacement of the target
                        
                    speed = self.gain * percent_offset_x # 根据比例设定速度
                    if speed < 0:
                        direction = -1
                    else:
                        direction = 1
                    self.move_cmd.angular.z = -direction * max(self.min_rotation_speed,
                                                min(self.max_rotation_speed, abs(speed)))  
                    self.move_cmd.linear.x = 0 # 如果不在一定范围内，线速度都设定为0
            # If we have depth values...
            else:   # 在一定范围内，才可以运行
                self.move_cmd.angular.z = 0
                if depth_set:
                    depth_set.sort()
                    # 框内平均深度
                    mean_z = float(sum(depth_set[:len(depth_set)/2])) / (len(depth_set)/2) # 取最近的一半的点数取平均

                    # Don't let the mean fall below the minimum reliable range
                    mean_z = max(self.min_z, mean_z)    # 设定深度不能比最小值大
                    print('The depth is ' + str(mean_z))
                    # Check the mean against the minimum range
                    '''if mean_z > self.min_z:
                        # Check the max range and goal threshold
                        if mean_z < self.max_z and (abs(mean_z - self.goal_z) > self.z_threshold):
                            speed = (mean_z - self.goal_z) * self.z_scale
                            self.move_cmd.linear.x = copysign(min(self.max_linear_speed, max(self.min_linear_speed, abs(speed))), speed)                  
                            print mean_z,speed      
                        else:
                             # Otherwise stop the robot
                             self.move_cmd.linear.x=0 '''
                    if abs(mean_z - self.goal_z) > self.z_threshold:     # 如果相对于目标在预定范围外，才会运动
                         speed = (mean_z - self.goal_z) * self.z_scale
                         if speed < 0:   # 后退时，速度可以快一些
                             speed *= 1.5
                             self.move_cmd.linear.x = copysign(min(self.max_linear_speed, max(self.min_linear_speed, abs(speed))), speed)
                         else:
                             self.move_cmd.linear.x = 0.0

        
        finally:
            # Release the lock
            self.lock.release()

    def convert_depth_image(self, ros_image):
        # Use cv_bridge() to convert the ROS image to OpenCV format
        try:
            # The depth image is a single-channel float32 image
            depth_image = self.cv_bridge.imgmsg_to_cv2(ros_image, "32FC1")
        except CvBridgeError, e:
            print e

        # Convert the depth image to a Numpy array
        self.depth_array = np.array(depth_image, dtype=np.float32)

    def get_camera_info(self, msg):
        self.image_width = msg.width
        self.image_height = msg.height

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        # Unregister the subscriber to stop cmd_vel publishing
        self.depth_subscriber.unregister()
        rospy.sleep(1)
        # Send an emtpy Twist message to stop the robot
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)      

if __name__ == '__main__':
    try:
        ObjectFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Object follower node terminated.")

