#!/usr/bin/env python3

import time
import numpy as np

from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge


import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

from turtlesim.msg import Pose

class SimpleController():

    def __init__(self):
        rospy.init_node('simple_controller', anonymous=True)
        rospy.on_shutdown(self.shutdown)


        self.cmd_fl_wheel = rospy.Publisher('/omni_control/front_left_wheel_joint/command',  Float64, queue_size=1)
        self.cmd_fr_wheel = rospy.Publisher('/omni_control/front_right_wheel_joint/command', Float64, queue_size=1)
        self.cmd_bl_wheel = rospy.Publisher('/omni_control/back_left_wheel_joint/command',   Float64, queue_size=1)
        self.cmd_br_wheel = rospy.Publisher('/omni_control/back_right_wheel_joint/command',  Float64, queue_size=1)

        subscriber = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        self.rate = rospy.Rate(30)

        self.v_x = 0
        self.v_y = 0
        self.w_z = 0
        self.wheel_vel = np.zeros(4)

    def cmd_vel_callback(self, msg:Twist):
        rospy.loginfo("cmd vel command: {0} ".format(msg))

        self.v_x = msg.linear.x
        self.v_y = msg.linear.y
        self.w_z = msg.angular.z

    def spin(self):
        while not rospy.is_shutdown():
            # calculate the commands for wheels and publish

            vw = np.array([self.v_x, self.v_y, self.w_z])
            # wheel radius
            r_w = 0.1016
            # Base geometry:
            l_x = 0.466725 / 2
            l_y = 0.2667 / 2

            # Inverse kinematic solution:
            w = 1/r_w * np.array([
                [1,  -1,  -(l_x + l_y)],
                [1,   1,   (l_x + l_y)],
                [1,   1,  -(l_x + l_y)],
                [1,  -1,   (l_x + l_y)]
            ])

            # There's OUTPUT:
            wheel_vel = w.dot(vw)

            self.cmd_fl_wheel.publish(wheel_vel[0])
            self.cmd_fr_wheel.publish(wheel_vel[1])
            self.cmd_bl_wheel.publish(wheel_vel[2])
            self.cmd_br_wheel.publish(wheel_vel[3])

            self.rate.sleep()

    def shutdown(self):
        self.cmd_fl_wheel.publish(0.0)
        self.cmd_fr_wheel.publish(0.0)
        self.cmd_bl_wheel.publish(0.0)
        self.cmd_br_wheel.publish(0.0)
        rospy.sleep(1)


simple_mover = SimpleController()
simple_mover.spin()