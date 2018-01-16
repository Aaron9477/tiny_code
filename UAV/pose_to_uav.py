#!/usr/bin/env python
import rospy
import time
# from demo_test.srv import *
# import nav_msgs
from geometry_msgs.msg import PoseStamped


class turtlebot_pos(object):
    def __init__(self, node_name):
        self.node_name = node_name

        rospy.init_node(node_name)
        rospy.loginfo("Starting node " + str(node_name))

        #rospy.on_shutdown(self.cleanup)

        # rospy.wait_for_service('teleop_ctrl_service2')
        # teleop_srv_init2 = rospy.ServiceProxy('teleop_ctrl_service2',teleop_ctrl)


        self.image_sub = rospy.Subscriber("/slam_out_pose", PoseStamped, self.odom_callback)


    def odom_callback(self, data):
        # move
        print 'move once'

        #print data.pose.position.x
        #print data.pose.position.y
        resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                               hover_pos_x = data.pose.pose.position.x,
                               hover_pos_y = - data.pose.pose.position.y,    # y is reverse
                               hover_pos_z = -1.2,
                               hover_pos_yaw = -1.57)
        print resp





if __name__ == "__main__":
    print "start test teleop control service"
    node_name = "get_odom"
    turtlebot_pos(node_name)
    rospy.spin()
