#!/usr/bin/env python3
# block: constant forward publisher

import rospy
from geometry_msgs.msg import Twist

rospy.init_node('forward_publisher')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
rate = rospy.Rate(10)                     # 10 Hz

msg = Twist()
msg.linear.x = 0.2                        # +X → вперёд
msg.angular.z = 0.0                       # без поворота

while not rospy.is_shutdown():
    pub.publish(msg)
    rate.sleep()