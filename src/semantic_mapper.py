#!/usr/bin/env python3
# -----------------------------------------------------------
# Build a semantic occupancy grid:
#   1) listen to /map (Hector SLAM)            -> nav_msgs/OccupancyGrid
#   2) listen to /darknet_ros/bounding_boxes   -> vision_msgs/Detection2DArray
#   3) project each detection centre onto the ground plane (assume z = 0)
#   4) paint the corresponding map cell with the detection class-id
#   5) publish SemanticGrid + colour preview
# -----------------------------------------------------------
import rospy, tf2_ros, tf2_geometry_msgs
import numpy as np, cv2
from nav_msgs.msg          import OccupancyGrid
from semantic_map_slam.msg import SemanticGrid
from vision_msgs.msg       import Detection2DArray
from geometry_msgs.msg     import PointStamped
from cv_bridge             import CvBridge

class SemanticMapper:
    def __init__(self):
        rospy.init_node('semantic_mapper')

        # parameters -------------------------------------------------
        self.base_frame  = rospy.get_param('~base_frame',  'base_front')
        self.camera_frame= rospy.get_param('~camera_frame','camera')
        self.map_topic   = rospy.get_param('~map_topic',   '/map')
        self.det_topic   = rospy.get_param('~det_topic',   
                                           '/darknet_ros/detections')
        self.out_topic   = rospy.get_param('~out_topic',   '/semantic_grid')
        # tools ------------------------------------------------------
        self.bridge  = CvBridge()
        self.tf_buf  = tf2_ros.Buffer()
        self.tf_li   = tf2_ros.TransformListener(self.tf_buf)

        # state ------------------------------------------------------
        self.map_msg     = None               # last OccupancyGrid
        self.class_grid  = None               # uint8 matrix of class-ids
        self.class_table = []                 # unique class-ids

        # pubs/subs --------------------------------------------------
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.cb_map, 
                         queue_size=1)
        rospy.Subscriber(self.det_topic, Detection2DArray, self.cb_det,
                         queue_size=5)
        self.pub = rospy.Publisher(self.out_topic, SemanticGrid, 
                                   queue_size=1)
        rospy.loginfo("Semantic mapper ready")
        rospy.spin()

    # ------------ callbacks ----------------------------------------
    def cb_map(self, msg):
        self.map_msg = msg
        w, h = msg.info.width, msg.info.height
        if self.class_grid is None:
            self.class_grid  = np.zeros((h, w), dtype=np.uint8)

    def cb_det(self, msg):
        if self.map_msg is None:                             # no map yet
            return
        # build lookup once per frame -------------------------------
        map_info = self.map_msg.info
        origin   = map_info.origin.position
        res      = map_info.resolution
        tf_ok = self.tf_buf.can_transform(
                    self.map_msg.header.frame_id,
                    msg.header.frame_id,
                    rospy.Time(0), rospy.Duration(0.1))
        if not tf_ok:
            rospy.logwarn_throttle(5,"TF missing map <-> camera")
            return

        # iterate detections ----------------------------------------
        for det in msg.detections:
            cx = det.bbox.center.x
            cy = det.bbox.center.y
            pt_cam = PointStamped()
            pt_cam.header = msg.header
            pt_cam.point.x, pt_cam.point.y, pt_cam.point.z = 0,0,0

            # project pixel to ray (simplified: take camera origin) --
            try:
                pt_map = tf2_geometry_msgs.do_transform_point(
                             pt_cam,
                             self.tf_buf.lookup_transform(
                                 self.map_msg.header.frame_id,
                                 msg.header.frame_id,
                                 rospy.Time(0)))
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                continue

            # map indices -------------------------------------------
            mx = int((pt_map.point.x - origin.x) / res)
            my = int((pt_map.point.y - origin.y) / res)
            if 0 <= mx < map_info.width and 0 <= my < map_info.height:
                class_id = det.results[0].id
                self.class_grid[my, mx] = class_id
                if class_id not in self.class_table:
                    self.class_table.append(class_id)

        self.publish_grid()

    # ----------- build and send SemanticGrid ------------------------
    def publish_grid(self):
        sg = SemanticGrid()
        sg.header.stamp = rospy.Time.now()
        sg.header.frame_id = self.map_msg.header.frame_id
        sg.grid  = self.map_msg
        sg.class_ids = self.class_table

        # colour preview --------------------------------------------
        colormap = np.zeros((256,3), np.uint8)
        np.random.seed(0)
        colormap[1:] = np.random.randint(0,255,(255,3))
        rgb = colormap[self.class_grid]
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        sg.preview = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        self.pub.publish(sg)

if __name__ == '__main__':
    SemanticMapper()