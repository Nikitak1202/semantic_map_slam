#!/usr/bin/env python3
"""
semantic_mapper.py
Publish RViz markers at map–frame positions where YOLO (darknet_ros) detects
objects and merge nearby markers into a single, “confirmed” detection.

A detection is accepted only if
  • its bounding–box centre is close enough to the optical axis (CENTER_TOLERANCE)
  • the corresponding lidar range is finite and shorter than MAX_MAPPING_RANGE
Supported classes: chair, refrigerator, sofa.
"""

import math, itertools, collections
from typing import Dict, Set, List

import rospy, tf2_ros, tf2_geometry_msgs, numpy as np
from std_msgs.msg          import ColorRGBA
from darknet_ros_msgs.msg  import BoundingBoxes
from sensor_msgs.msg       import LaserScan, CameraInfo
from nav_msgs.msg          import OccupancyGrid
from geometry_msgs.msg     import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# ---------- user-tunable ------------------------------------------------------
SUPPORTED_CLASSES   = ["chair", "refrigerator", "sofa"]
CENTER_TOLERANCE    = 0.3          # bbox centre ±20 % of image half-width
MAX_MAPPING_RANGE   = 6           # ignore detections farther than this [m]

CLUSTER_DIST           = 1      # markers ≤ 0.50 m apart belong together
CLUSTER_PERIOD         = 2        # check clusters every 2 s
CLUSTER_CONFIRM_ITERS  = 3          # need 3 consecutive checks to collapse

COLORS = { "chair":(0.10,0.80,0.10,0.85),
           "refrigerator":(0.10,0.10,0.80,0.85),
           "sofa":(0.80,0.10,0.10,0.85) }
SIZE_M = { "chair":0.45, "refrigerator":0.60, "sofa":0.75 }

MARKER_LIFETIME = 0.0               # 0 ⇒ keep forever
CAMERA_FRAME    = "base_front"
# -----------------------------------------------------------------------------


class SemanticMapper:
    # -------------------------------------------------------------------------
    def __init__(self):
        rospy.init_node("semantic_mapper")

        # publishers / subscribers
        self.pub_markers = rospy.Publisher("semantic_map_markers",
                                           MarkerArray, queue_size=10, latch=True)

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
        self.marker_id   = itertools.count()      # unique IDs
        self.img_w, self.hfov = 640, math.radians(90)  # until CameraInfo arrives

        # camera info (one-shot)
        rospy.Subscriber("/head/camera/camera_info", CameraInfo,
                         self.cb_caminfo, queue_size=1)

        # ----- NEW: book-keeping for clustering ------------------------------
        self.records: Dict[int, dict] = {}        # marker-id → info dict
        self.cluster_persist: Dict[frozenset, int] = {}

        rospy.Timer(rospy.Duration(CLUSTER_PERIOD), self.timer_cluster)

        rospy.loginfo("semantic_mapper ready")

    # -------------------------------------------------------------------------
    def cb_caminfo(self, msg: CameraInfo):
        self.img_w = msg.width
        fx         = msg.K[0]
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

        m_arr = MarkerArray()
        stamp_now = rospy.Time.now()

        for box in msg.bounding_boxes:
            cls = box.Class.lower()
            if cls not in SUPPORTED_CLASSES:
                continue

            # centrality gate --------------------------------------------------
            u_mid = 0.5 * (box.xmin + box.xmax)
            if abs(u_mid - self.img_w/2.0) > CENTER_TOLERANCE * (self.img_w/2.0):
                continue

            # pixel → bearing angle -------------------------------------------
            theta = ((u_mid - self.img_w/2.0) / self.img_w) * self.hfov

            # nearest lidar ray & range ---------------------------------------
            idx = int(round((theta - angle_min) / angle_inc))
            if not (0 <= idx < n_beams):
                continue
            rng = ranges[idx]
            if not (math.isfinite(rng) and 0.0 < rng <= MAX_MAPPING_RANGE):
                continue

            # point in camera frame -------------------------------------------
            pt_cam = PointStamped()
            pt_cam.header.stamp    = self.latest_scan.header.stamp
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.point.x = rng * math.cos(theta)
            pt_cam.point.y = rng * math.sin(theta)
            pt_cam.point.z = 0.0

            # transform → map --------------------------------------------------
            try:
                pt_map = self.tf_buf.transform(pt_cam, self.map_frame,
                                               rospy.Duration(0.2))
            except (tf2_ros.LookupException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                rospy.logwarn_throttle(
                    2.0, f"TF {self.map_frame}←{CAMERA_FRAME} unavailable")
                continue

            # marker -----------------------------------------------------------
            new_marker = self.make_marker(cls, pt_map.point, stamp_now)
            m_arr.markers.append(new_marker)

            # remember it for clustering
            self.records[new_marker.id] = dict(
                class_=cls,
                pos=np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
            )

        if m_arr.markers:
            self.pub_markers.publish(m_arr)

    # -------------------------------------------------------------------------
    def timer_cluster(self, _):
        """Periodically check clusters and collapse persistent ones."""
        ids   = list(self.records.keys())
        N     = len(ids)
        if N < 2:
            self.cluster_persist.clear()          # nothing to cluster
            return

        # ---- build distance matrix & breadth-first grouping -----------------
        positions = np.array([self.records[i]["pos"] for i in ids])
        clusters: List[Set[int]] = []
        unvisited: Set[int] = set(range(N))       # indices in 'ids' list

        while unvisited:
            root = unvisited.pop()
            stack = [root]
            cluster = {root}
            while stack:
                i = stack.pop()
                # find neighbours of 'i' inside threshold
                dists = np.linalg.norm(positions[list(unvisited)] - positions[i], axis=1)
                neighbours = [j for j, d in zip(list(unvisited), dists) if d <= CLUSTER_DIST]
                for j in neighbours:
                    unvisited.remove(j)
                    stack.append(j)
                    cluster.add(j)
            clusters.append({ids[k] for k in cluster})   # convert to real marker IDs

        # ---- update persistence counters & collapse if due ------------------
        new_persist = {}
        for cl in clusters:
            if len(cl) == 1:                 # singleton → no clustering
                continue
            key = frozenset(cl)
            cnt = self.cluster_persist.get(key, 0) + 1
            if cnt >= CLUSTER_CONFIRM_ITERS:
                self.collapse_cluster(cl)    # replaces markers
            else:
                new_persist[key] = cnt       # keep counting

        self.cluster_persist = new_persist   # forget vanishing clusters

    # -------------------------------------------------------------------------
    def collapse_cluster(self, ids: Set[int]):
        """Replace all markers in *ids* with one averaged marker."""
        # gather classes & positions
        classes = [self.records[i]["class_"] for i in ids]
        positions = np.array([self.records[i]["pos"] for i in ids])

        centroid = positions.mean(axis=0)
        # majority class
        majority = max(classes, key=classes.count)

        # delete old markers --------------------------------------------------
        delete_arr = MarkerArray()
        for mid in ids:
            old_cls = self.records[mid]["class_"]
            del self.records[mid]            # remove from records

            m_del = Marker()
            m_del.header.frame_id = self.map_frame
            m_del.header.stamp    = rospy.Time.now()
            m_del.ns  = old_cls
            m_del.id  = mid
            m_del.action = Marker.DELETE
            delete_arr.markers.append(m_del)

        # publish deletions in one shot
        if delete_arr.markers:
            self.pub_markers.publish(delete_arr)

        # add averaged marker -------------------------------------------------
        pt = PointStamped()
        pt.point.x, pt.point.y, pt.point.z = centroid
        new_marker = self.make_marker(majority, pt.point, rospy.Time.now())

        add_arr = MarkerArray();
        add_arr.markers.append(new_marker)
        self.pub_markers.publish(add_arr)

        # remember new marker
        self.records[new_marker.id] = dict(class_=majority, pos=centroid)

    # -------------------------------------------------------------------------
    def make_marker(self, cls_name, pt_map, stamp):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = stamp
        m.ns   = cls_name
        m.id   = next(self.marker_id)
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
