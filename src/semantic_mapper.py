#!/usr/bin/env python3
"""
semantic_mapper.py
Publish RViz markers at map–frame positions where YOLO (darknet_ros) detects objects.
A detection is accepted only if
  • its bounding–box centre is close enough to the optical axis (CENTER_TOLERANCE)
  • the corresponding lidar range is finite and shorter than MAX_MAPPING_RANGE
Supported classes: chair, refrigerator, sofa.
"""

import math, itertools
from collections import deque

import rospy, tf2_ros, tf2_geometry_msgs
from std_msgs.msg          import ColorRGBA
from darknet_ros_msgs.msg  import BoundingBoxes
from sensor_msgs.msg       import LaserScan, CameraInfo
from nav_msgs.msg          import OccupancyGrid
from geometry_msgs.msg     import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# ---------- user-tunable ------------------------------------------------------
SUPPORTED_CLASSES = ["chair", "refrigerator", "sofa"]
CENTER_TOLERANCE  = 0.2          # 0.20 ⇒ bbox centre ±20 % of image half-width
MAX_MAPPING_RANGE = 3           # [m] ignore detections farther than this

COLORS = { "chair":(0.10,0.80,0.10,0.85),
           "refrigerator":(0.10,0.10,0.80,0.85),
           "sofa":(0.80,0.10,0.10,0.85) }
SIZE_M = { "chair":0.45, "refrigerator":0.60, "sofa":0.75 }

MARKER_LIFETIME = 0.0             # 0 ⇒ keep forever
CAMERA_FRAME    = "base_front"
# -----------------------------------------------------------------------------


class SemanticMapper:
    # -------------------------------------------------------------------------
    def __init__(self):
        rospy.init_node("semantic_mapper")

        # publishers / subscribers
        self.pub_markers = rospy.Publisher("semantic_map_markers",
                                           MarkerArray, queue_size=1, latch=True)

        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes,
                         self.cb_boxes, queue_size=5)
        rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=5)
        rospy.Subscriber("/map", OccupancyGrid, self.cb_map, queue_size=1)

        # TF listener
        self.tf_buf = tf2_ros.Buffer(rospy.Duration(30))
        tf2_ros.TransformListener(self.tf_buf)

        # state
        self.latest_scan = None
        self.map_frame   = "map"
        self.marker_seq  = itertools.count()      # unique IDs
        self.img_w, self.hfov = 640, math.radians(90)  # until CameraInfo arrives

        # one-shot CameraInfo
        rospy.Subscriber("/head/camera/camera_info", CameraInfo,
                         self.cb_caminfo, queue_size=1)

        rospy.loginfo("semantic_mapper ready")

    # -------------------------------------------------------------------------
    def cb_caminfo(self, msg: CameraInfo):
        self.img_w = msg.width
        fx         = msg.K[0]                     # focal length (px)
        if fx > 0:
            self.hfov = 2.0 * math.atan2(0.5*self.img_w, fx)
        rospy.loginfo_once(f"CameraInfo: width={self.img_w}, "
                           f"HFOV={math.degrees(self.hfov):.1f}°")

    def cb_map(self, msg: OccupancyGrid):
        self.map_frame = msg.header.frame_id or "map"

    def cb_scan(self, msg: LaserScan):
        self.latest_scan = msg

    # -------------------------------------------------------------------------
    def cb_boxes(self, msg: BoundingBoxes):
        if self.latest_scan is None:
            return

        angle_min, angle_inc = self.latest_scan.angle_min, self.latest_scan.angle_increment
        ranges = self.latest_scan.ranges
        n_beams = len(ranges)

        marker_array = MarkerArray()
        timestamp    = rospy.Time.now()

        for box in msg.bounding_boxes:
            cls = box.Class.lower()
            if cls not in SUPPORTED_CLASSES:
                continue

            # ---------- centrality check -------------------------------------
            u_mid   = 0.5 * (box.xmin + box.xmax)             # pixel column
            if abs(u_mid - self.img_w/2.0) > CENTER_TOLERANCE * (self.img_w/2.0):
                continue

            # ---------- pixel → bearing angle --------------------------------
            theta = ((u_mid - self.img_w/2.0) / self.img_w) * self.hfov

            # ---------- nearest lidar beam & range ---------------------------
            idx = int(round((theta - angle_min) / angle_inc))
            if not (0 <= idx < n_beams):
                continue
            rng = ranges[idx]
            if not (math.isfinite(rng) and 0.0 < rng <= MAX_MAPPING_RANGE):
                continue                        # fails range gate

            # ---------- point in camera frame --------------------------------
            pt_cam = PointStamped()
            pt_cam.header.stamp    = self.latest_scan.header.stamp
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.point.x = rng * math.cos(theta)
            pt_cam.point.y = rng * math.sin(theta)
            pt_cam.point.z = 0.0

            # ---------- transform → map frame --------------------------------
            try:
                pt_map = self.tf_buf.transform(pt_cam, self.map_frame,
                                               rospy.Duration(0.2))
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                rospy.logwarn_throttle(
                    2.0, f"TF {self.map_frame}←{CAMERA_FRAME} unavailable")
                continue

            # ---------- marker ----------------------------------------------
            marker_array.markers.append(
                self.make_marker(cls, pt_map.point, timestamp))

        if marker_array.markers:
            self.pub_markers.publish(marker_array)

    # -------------------------------------------------------------------------
    def make_marker(self, cls_name, pt_map, stamp):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = stamp
        m.ns   = cls_name
        m.id   = next(self.marker_seq)
        m.type = Marker.CUBE
        m.action = Marker.ADD

        m.pose.position.x = pt_map.x
        m.pose.position.y = pt_map.y
        m.pose.position.z = pt_map.z
        m.pose.orientation.w = 1.0

        r,g,b,a = COLORS.get(cls_name, (1,1,0,0.9))
        m.color = ColorRGBA(r,g,b,a)
        s = SIZE_M.get(cls_name, 0.4)
        m.scale.x = m.scale.y = m.scale.z = s
        m.lifetime = rospy.Duration(MARKER_LIFETIME)
        return m


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        SemanticMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
