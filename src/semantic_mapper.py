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
from sensor_msgs.msg import LaserScan, CameraInfo
from nav_msgs.msg  import OccupancyGrid
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# ---------- user-tunable -----------------------------------------------------
SUPPORTED_CLASSES = ["chair", "refrigerator", "sofa"]

COLORS = {          #  R,   G,   B,   A
    "chair"        : (0.10, 0.80, 0.10, 0.85),
    "refrigerator" : (0.10, 0.10, 0.80, 0.85),
    "sofa"         : (0.80, 0.10, 0.10, 0.85),
}
SIZE_M = {  # cube edge length [m]
    "chair"        : 0.45,
    "refrigerator" : 0.60,
    "sofa"         : 0.75,
}

MARKER_LIFETIME = 0.0           # 0 ⇒ keep forever
CAMERA_FRAME    = "base_front"       # where the camera lives

# ---- NEW: how “central” a detection must be to count -----------------------
# expressed as fraction of the half–image-width (0.0 … 1.0)
CENTER_TOLERANCE = 0.5     # 0.10 ⇒ accept if bbox centre is within ±10 % of
                            #         the image centre.  Lower = stricter.
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

        # camera info
        self._init_camera_info()    # subscribe to CameraInfo

        rospy.loginfo("semantic_mapper ready")

    # ---------------------------------------------------------------------------
    #  CameraInfo handling — paste this right after __init__()
    # ---------------------------------------------------------------------------
    def _init_camera_info(self):
        """Subscribe once to CameraInfo, grab width and HFOV."""
        self.img_w  = 640            # fallback until we hear CameraInfo
        self.hfov   = math.radians(90)  # fallback HFOV in radians

        self.caminfo_sub = rospy.Subscriber("/head/camera/camera_info", CameraInfo,
                        self.cb_caminfo, queue_size=1)

    def cb_caminfo(self, msg):
        """Store real image width and horizontal FoV."""
        self.img_w = msg.width
        fx         = msg.K[0]        # focal length (pixels)
        if fx > 0:
            self.hfov = 2.0 * math.atan2(0.5 * self.img_w, fx)
        # unsubscribe – we only need one message
        rospy.loginfo_once(f"Camera info: width={self.img_w}, hfov={math.degrees(self.hfov):.1f}°")
        self.caminfo_sub.unregister()


    def cb_map(self, msg):
        self.map_frame = msg.header.frame_id or "map"

    def cb_scan(self, msg):
        self.latest_scan = msg

    def cb_boxes(self, msg):
        """
        Handle a darknet_ros BoundingBoxes message:
        turn every *central* detection into a map-frame Marker.
        """
        if self.latest_scan is None:
            return                      # waiting for first /scan

        self.img_w      = 640                          # darknet default
        angle_inc  = self.latest_scan.angle_increment
        angle_min  = self.latest_scan.angle_min

        marker_array = MarkerArray()
        stamp_now    = rospy.Time.now()

        for box in msg.bounding_boxes:
            cls = box.Class.lower()
            if cls not in SUPPORTED_CLASSES:
                continue

            # ------------------------------------------------------------------
            # 1.  Centre-of-bbox must be close enough to centre of the image
            # ------------------------------------------------------------------
            x_mid = 0.5 * (box.xmin + box.xmax)      # [px]
            offset_px = abs(x_mid - self.img_w / 2.0)     # distance from centre [px]
            if offset_px > CENTER_TOLERANCE * (self.img_w / 2.0):
                continue            # too far from centre ⇒ skip this detection

            # 2.  Convert pixel offset ► bearing angle in CAMERA_FRAME
            rel   = (x_mid - self.img_w / 2.0) / self.img_w     # -0.5 … +0.5
            theta = rel * self.hfov

            # 3.  Pick the nearest laser ray
            idx = int(round((theta - angle_min) / angle_inc))
            if not (0 <= idx < len(self.latest_scan.ranges)):
                continue
            rng = self.latest_scan.ranges[idx]
            if not math.isfinite(rng):
                continue

            # 4.  Build point in CAMERA_FRAME
            pt_cam              = PointStamped()
            pt_cam.header.stamp = self.latest_scan.header.stamp
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.point.x = rng * math.cos(theta)
            pt_cam.point.y = rng * math.sin(theta)
            pt_cam.point.z = 0.0

            # 5.  Transform to MAP (latest available transform)
            pt_cam.header.stamp = rospy.Time(0)
            try:
                pt_map = self.tf_buffer.transform(
                    pt_cam, self.map_frame, rospy.Duration(0.2))
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                rospy.logwarn_throttle(
                    2.0, f"TF map←{CAMERA_FRAME} unavailable – skipping {cls}")
                continue

            # 6.  Create & append marker
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
