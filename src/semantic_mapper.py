#!/usr/bin/env python3
"""
semantic_mapper.py
Publish RViz markers at map–frame positions where YOLO (darknet_ros) detects objects.
Supported classes: chair, refrigerator, sofa.

@author Nikita Kudriavtsev
"""

import math
import itertools
from collections import deque
from std_msgs.msg          import ColorRGBA
import rospy
import tf2_ros
import tf2_geometry_msgs

from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import LaserScan
from nav_msgs.msg  import OccupancyGrid
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# ----- user-tunable ----------------------------------------------------------
SUPPORTED_CLASSES = ["chair", "refrigerator", "sofa"]
COLORS = {  # RGBA
    "chair"        : (0.2, 0.6, 1.0, 0.9),
    "refrigerator" : (1.0, 1.0, 1.0, 0.9),
    "sofa"         : (0.9, 0.3, 0.3, 0.9),
}
MARKER_SCALE = 0.3                # [m] cube edge
CAMERA_FOV   = math.radians(90)   # horizontal FoV of the darknet camera
CAMERA_FRAME = "base_front"       # where the camera lives
# -----------------------------------------------------------------------------


class SemanticMapper:
    def __init__(self):
        rospy.init_node("semantic_mapper")

        # i/o
        self.pub_markers = rospy.Publisher("semantic_map_markers",
                                           MarkerArray, queue_size=1, latch=True)
        rospy.Subscriber("/darknet_ros/bounding_boxes",
                         BoundingBoxes, self.cb_boxes, queue_size=5)
        rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=5)
        rospy.Subscriber("/map", OccupancyGrid, self.cb_map, queue_size=1)

        # tf
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30))
        tf2_ros.TransformListener(self.tf_buffer)

        # state
        self.latest_scan = None
        self.marker_seq = 0
        self.map_frame   = "map"      # default, updated from /map
        self.id_counter  = itertools.count(0)  # unique marker IDs

        # keep only the most recent detections to avoid flooding markers
        self.history = {cls: deque(maxlen=20) for cls in SUPPORTED_CLASSES}

        rospy.loginfo("semantic_mapper ready")

    # ---------------------------------------------------------------------

    def cb_map(self, msg):
        self.map_frame = msg.header.frame_id or "map"

    def cb_scan(self, msg):
        self.latest_scan = msg

    def cb_boxes(self, msg):
        """
        Handle a darknet_ros BoundingBoxes message:
        turn every recognised object into a map-frame Marker.
        """
        if self.latest_scan is None:
            return                      # not yet ready

        img_w      = 640
        angle_inc  = self.latest_scan.angle_increment
        angle_min  = self.latest_scan.angle_min

        marker_array = MarkerArray()
        stamp_now    = rospy.Time.now()

        for box in msg.bounding_boxes:
            cls = box.Class.lower()
            if cls not in SUPPORTED_CLASSES:
                continue

            # 1. camera pixel  ➜  bearing angle
            x_mid   = 0.5 * (box.xmin + box.xmax)
            rel     = (x_mid - img_w / 2.0) / img_w          # -0.5 … +0.5
            theta   = rel * CAMERA_FOV

            # 2. pick closest laser ray
            idx = int(round((theta - angle_min) / angle_inc))
            if not (0 <= idx < len(self.latest_scan.ranges)):
                continue
            rng = self.latest_scan.ranges[idx]
            if not math.isfinite(rng):
                continue

            # 3. build a point in CAMERA_FRAME (== base_front)
            pt_cam              = PointStamped()
            pt_cam.header.stamp = self.latest_scan.header.stamp
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.point.x = rng * math.cos(theta)
            pt_cam.point.y = rng * math.sin(theta)
            pt_cam.point.z = 0.0

            # 4. transform to MAP using the **newest available** TF
            pt_cam.header.stamp = rospy.Time(0)          # 0 = latest transform
            try:
                pt_map = self.tf_buffer.transform(
                    pt_cam,               # message to transform
                    self.map_frame,       # target frame  ("/map")
                    rospy.Duration(0.2))  # wait max 0.2 s
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                rospy.logwarn_throttle(
                    2.0,
                    f"TF map←{CAMERA_FRAME} unavailable – skipping {cls}")
                continue


            # 5. cook & add marker
            marker = self.make_marker(cls, pt_map.point, stamp_now)
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.pub_markers.publish(marker_array)


    def make_marker(self, cls_name, pt_map, stamp):
        """
        Build a single RViz marker fixed in the map frame.

        Args
        ----
        cls_name : str
            Detected object class ("chair", "sofa", "refrigerator", …)
        pt_map   : geometry_msgs/Point
            3-D point already expressed in the /map frame.
        stamp    : rospy.Time
            Timestamp to place in the marker header.
        """
        marker = Marker()
        marker.id = self.marker_seq
        self.marker_seq += 1

        # ------------------------------------------------------------------
        # Marker header – tie it to the *static* map frame so it never moves
        # ------------------------------------------------------------------
        marker.header.frame_id = self.map_frame          # usually "map"
        marker.header.stamp    = stamp                   # <<< use passed stamp

        # Namespace/ID let RViz update the same object instead of redrawing it
        marker.ns = cls_name
        marker.id = self.marker_seq
        self.marker_seq += 1                              # keep unique IDs

        marker.type   = Marker.CUBE
        marker.action = Marker.ADD

        # ---------------------------------------------------
        # Pose: coordinates are already in /map
        # ---------------------------------------------------
        marker.pose.position.x = pt_map.x
        marker.pose.position.y = pt_map.y
        marker.pose.position.z = pt_map.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0  # neutral orientation

        # ---------------------------------------------------
        # Class-specific appearance
        # ---------------------------------------------------
        colour_table = {
            "chair"        : (0.10, 0.80, 0.10, 0.85),  # green
            "sofa"         : (0.80, 0.10, 0.10, 0.85),  # red
            "refrigerator" : (0.10, 0.10, 0.80, 0.85),  # blue
        }
        size_table = {
            "chair"        : 0.45,
            "sofa"         : 0.75,
            "refrigerator" : 0.60,
        }

        rgba = colour_table.get(cls_name, (0.95, 0.95, 0.10, 0.85))  # default yellow
        size = size_table.get(cls_name, 0.40)                         # default size

        marker.color = ColorRGBA(*rgba)
        marker.scale.x = size
        marker.scale.y = size
        marker.scale.z = size

        marker.lifetime = rospy.Duration(0)  # live forever

        return marker


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        SemanticMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
