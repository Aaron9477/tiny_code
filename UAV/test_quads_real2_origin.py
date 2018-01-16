#!/usr/bin/env python
import rospy
import time
from demo_test.srv import *

def test_server_fun():
    # init service
    rospy.wait_for_service('teleop_ctrl_service2')
    rospy.wait_for_service('teleop_ctrl_service3')
    rospy.wait_for_service('teleop_ctrl_service4')
    teleop_srv_init2 = rospy.ServiceProxy('teleop_ctrl_service2',teleop_ctrl)
    teleop_srv_init3 = rospy.ServiceProxy('teleop_ctrl_service3',teleop_ctrl)
    teleop_srv_init4 = rospy.ServiceProxy('teleop_ctrl_service4',teleop_ctrl)

    # takeoff
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_ARM_DISARM,
                           base_contrl = teleop_ctrlRequest.ARM_TAKEOFF)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_ARM_DISARM,
                           base_contrl = teleop_ctrlRequest.ARM_TAKEOFF)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_ARM_DISARM,
                           base_contrl = teleop_ctrlRequest.ARM_TAKEOFF)
    print resp
    time.sleep(10)

    # fly to point 1
    print 'stage 1'
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 1.0,
                           hover_pos_y = 1.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = -1.0,
                           hover_pos_y = 1.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = -1.414,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    time.sleep(5)
    
    # fly to point 2
    print 'stage 2'
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = -1.0,
                           hover_pos_y = -1.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 1.0,
                           hover_pos_y = -1.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = 1.414,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    time.sleep(20)

    # fly to point 3
    print 'stage 3'
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = -1.0,
                           hover_pos_y = -1.0,
                           hover_pos_z = -1.3,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 1.0,
                           hover_pos_y = -1.0,
                           hover_pos_z = -1.3,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = 1.414,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    time.sleep(3)

    # fly to point 4
    print 'stage 4'
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 1.0,
                           hover_pos_y = 1.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = -1.0,
                           hover_pos_y = 1.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = -1.414,
                           hover_pos_z = -1.3,
                           hover_pos_yaw = -1.57)
    print resp
    time.sleep(20)

    # fly to point 5
    print 'stage 5'
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = -1.2,
                           hover_pos_y = 0.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = 0.0)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = 0.0,
                           hover_pos_z = -1.3,
                           hover_pos_yaw = 0.0)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 1.2,
                           hover_pos_y = 0.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = 0.0)
    print resp
    time.sleep(13)

    # fly to point 6
    print 'stage 6'
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = 0.0,
                           hover_pos_z = -1.2,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = 1.2,
                           hover_pos_z = -1.3,
                           hover_pos_yaw = -1.57)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_HOVER_POS,
                           hover_pos_x = 0.0,
                           hover_pos_y = -1.2,
                           hover_pos_z = -1.3,
                           hover_pos_yaw = -1.57)
    print resp
    time.sleep(10)

    # land and disarm
    resp = teleop_srv_init2(teleop_ctrl_mask = teleop_ctrlRequest.MASK_ARM_DISARM,
                           base_contrl = teleop_ctrlRequest.LAND_DISARM)
    print resp
    resp = teleop_srv_init3(teleop_ctrl_mask = teleop_ctrlRequest.MASK_ARM_DISARM,
                           base_contrl = teleop_ctrlRequest.LAND_DISARM)
    print resp
    resp = teleop_srv_init4(teleop_ctrl_mask = teleop_ctrlRequest.MASK_ARM_DISARM,
                           base_contrl = teleop_ctrlRequest.LAND_DISARM)
    print resp

    print 'task done!'
    time.sleep(2)

    return resp


if __name__ == "__main__":
    print "start test teleop control service"
    test_server_fun()
    print 'exit!'
