#!/usr/bin/env python3
"""
semantic_mapper.py
Publishes RViz markers for YOLO detections and fuses nearby ones into
“confirmed” clusters.

Standalone (unconfirmed) markers disappear after each clustering cycle.
"""

import math, itertools
from typing import Dict, Set, List

import rospy, tf2_ros, tf2_geometry_msgs, numpy as np
from std_msgs.msg           import ColorRGBA
from darknet_ros_msgs.msg   import BoundingBoxes
from sensor_msgs.msg        import LaserScan, CameraInfo
from nav_msgs.msg           import OccupancyGrid
from geometry_msgs.msg      import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# ---------- user-tunable -----------------------------------------------------
SUPPORTED_CLASSES  = ["chair", "refrigerator", "sofa"]
CENTER_TOLERANCE   = 0.2          # bbox centre ±30 % of image half-width
MAX_MAPPING_RANGE  = 6          # ignore detections farther than this [m]

CLUSTER_DIST          = 1.0       # [m] – markers ≤1 m apart → same cluster
CLUSTER_PERIOD        = 2.0       # run clustering every 2 s
CLUSTER_CONFIRM_ITERS = 3         # collapse after 3 consecutive hits

COLORS = {
    "chair":        (0.10, 0.80, 0.10, 0.85),
    "refrigerator": (0.10, 0.10, 0.80, 0.85),
    "sofa":         (0.80, 0.10, 0.10, 0.85)
}
SIZE_M = {"chair": 0.45, "refrigerator": 0.60, "sofa": 0.75}

MARKER_LIFETIME = 0.0             # 0 ⇒ keep forever
CAMERA_FRAME    = "base_front"
# ---------------------------------------------------------------------------


class SemanticMapper:
    def __init__(self):
        rospy.init_node("semantic_mapper")

        # --- I/O ------------------------------------------------------------
        self.pub_markers = rospy.Publisher(
            "semantic_map_markers", MarkerArray, queue_size=10, latch=True)

        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes,
                         self.cb_boxes, queue_size=5)
        rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=5)
        rospy.Subscriber("/map",  OccupancyGrid, self.cb_map,  queue_size=1)

        # TF
        self.tf_buf = tf2_ros.Buffer(rospy.Duration(30))
        tf2_ros.TransformListener(self.tf_buf)

        # State
        self.latest_scan = None
        self.map_frame   = "map"
        self.marker_id   = itertools.count()
        self.img_w, self.hfov = 640, math.radians(90)

        rospy.Subscriber("/head/camera/camera_info", CameraInfo,
                         self.cb_caminfo, queue_size=1)

        # bookkeeping  (id → {class_, pos, confirmed})
        self.records: Dict[int, dict] = {}
        self.cluster_persist: Dict[frozenset, int] = {}

        rospy.Timer(rospy.Duration(CLUSTER_PERIOD), self.timer_cluster)
        rospy.loginfo("semantic_mapper ready")

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
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

    def cb_boxes(self, msg: BoundingBoxes):
        if self.latest_scan is None:
            return

        a_min, a_inc = self.latest_scan.angle_min, self.latest_scan.angle_increment
        ranges       = self.latest_scan.ranges
        N_beams      = len(ranges)

        m_arr   = MarkerArray()
        stamp   = rospy.Time.now()

        for box in msg.bounding_boxes:
            cls = box.Class.lower()
            if cls not in SUPPORTED_CLASSES:
                continue

            # centrality gate
            u_mid = 0.5 * (box.xmin + box.xmax)
            if abs(u_mid - self.img_w/2.0) > CENTER_TOLERANCE*(self.img_w/2.0):
                continue

            # pixel → bearing
            theta = ((u_mid - self.img_w/2.0)/self.img_w) * self.hfov

            # nearest beam
            idx = int(round((theta - a_min)/a_inc))
            if not (0 <= idx < N_beams):
                continue
            rng = ranges[idx]
            if not (math.isfinite(rng) and 0.0 < rng <= MAX_MAPPING_RANGE):
                continue

            # point in camera frame
            pt_cam = PointStamped()
            pt_cam.header.stamp    = self.latest_scan.header.stamp
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.point.x = rng*math.cos(theta)
            pt_cam.point.y = rng*math.sin(theta)
            pt_cam.point.z = 0.0

            # transform → map
            try:
                pt_map = self.tf_buf.transform(pt_cam, self.map_frame,
                                               rospy.Duration(0.2))
            except tf2_ros.TransformException:
                rospy.logwarn_throttle(
                    2.0, f"TF {self.map_frame}←{CAMERA_FRAME} unavailable")
                continue

            # create marker
            m = self.make_marker(cls, pt_map.point, stamp)
            m_arr.markers.append(m)

            # store record (unconfirmed)
            self.records[m.id] = dict(class_=cls,
                                      pos=np.array([pt_map.point.x,
                                                    pt_map.point.y,
                                                    pt_map.point.z]),
                                      confirmed=False)

        if m_arr.markers:
            self.pub_markers.publish(m_arr)

    # -----------------------------------------------------------------------
    # Clustering logic
    # -----------------------------------------------------------------------
    def timer_cluster(self, _):
        ids = list(self.records.keys())
        if not ids:        # nothing to do
            self.cluster_persist.clear()
            return

        # build clusters -----------------------------------------------------
        positions = np.array([self.records[i]["pos"] for i in ids])
        unvis = set(range(len(ids)))
        clusters: List[Set[int]] = []

        while unvis:
            root = unvis.pop()
            stack = [root]
            cl_idx = {root}
            while stack:
                i = stack.pop()
                d = np.linalg.norm(positions[list(unvis)] - positions[i], axis=1)
                neigh = [j for j, dist in zip(list(unvis), d) if dist <= CLUSTER_DIST]
                for j in neigh:
                    unvis.remove(j)
                    stack.append(j)
                    cl_idx.add(j)
            clusters.append({ids[k] for k in cl_idx})

        # process clusters ---------------------------------------------------
        new_persist: Dict[frozenset, int] = {}
        for cl in clusters:
            if len(cl) > 1:                                # multi-element
                key = frozenset(cl)
                cnt = self.cluster_persist.get(key, 0) + 1
                if cnt >= CLUSTER_CONFIRM_ITERS:
                    self.collapse_cluster(cl)
                else:
                    new_persist[key] = cnt
        self.cluster_persist = new_persist

        # delete unconfirmed singletons -------------------------------------
        del_arr = MarkerArray()
        for cl in clusters:
            if len(cl) == 1:
                mid = next(iter(cl))
                rec = self.records.get(mid)
                if rec and not rec["confirmed"]:
                    m_del = Marker()
                    m_del.header.frame_id = self.map_frame
                    m_del.header.stamp    = rospy.Time.now()
                    m_del.ns   = rec["class_"]
                    m_del.id   = mid
                    m_del.action = Marker.DELETE
                    del_arr.markers.append(m_del)
                    del self.records[mid]          # drop record
        if del_arr.markers:
            self.pub_markers.publish(del_arr)

    def collapse_cluster(self, ids: Set[int]):
        # gather info
        classes   = [self.records[i]["class_"] for i in ids]
        positions = np.array([self.records[i]["pos"]   for i in ids])
        centroid  = positions.mean(axis=0)
        majority  = max(classes, key=classes.count)

        # delete originals
        del_arr = MarkerArray()
        for mid in ids:
            old_cls = self.records[mid]["class_"]
            m_del = Marker()
            m_del.header.frame_id = self.map_frame
            m_del.header.stamp    = rospy.Time.now()
            m_del.ns   = old_cls
            m_del.id   = mid
            m_del.action = Marker.DELETE
            del_arr.markers.append(m_del)
            del self.records[mid]
        if del_arr.markers:
            self.pub_markers.publish(del_arr)

        # add averaged “confirmed” marker
        pt = PointStamped()
        pt.point.x, pt.point.y, pt.point.z = centroid
        new_m = self.make_marker(majority, pt.point, rospy.Time.now())

        add_arr = MarkerArray(markers=[new_m])
        self.pub_markers.publish(add_arr)

        self.records[new_m.id] = dict(class_=majority,
                                      pos=centroid,
                                      confirmed=True)

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------
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

        r, g, b, a = COLORS.get(cls_name, (1, 1, 0, 0.9))
        m.color = ColorRGBA(r, g, b, a)
        s       = SIZE_M.get(cls_name, 0.4)
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
