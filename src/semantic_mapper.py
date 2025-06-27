#!/usr/bin/env python3
"""
semantic_mapper.py
Publishes RViz markers for YOLO detections and fuses nearby ones into
“confirmed” clusters.
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


# ───────────────────────── USER TUNABLES ──────────────────────────
SUPPORTED_CLASSES         = ["chair", "refrigerator", "sofa"]
CENTER_TOLERANCE          = 0.4           # % half-width of image
MAX_MAPPING_RANGE         = 3.0           # m
CLUSTER_DIST              = 1.2           # m
CLUSTER_PERIOD            = 2.0           # s
CLUSTER_CONFIRM_ITERS     = 5             # consecutive hits
COLORS = {
    "chair":        (0.10, 0.80, 0.10, 0.85),
    "refrigerator": (0.10, 0.10, 0.80, 0.85),
    "sofa":         (0.80, 0.10, 0.10, 0.85),
}
SIZE_M = {"chair": 0.45, "refrigerator": 0.60, "sofa": 0.75}

MARKER_LIFETIME = 0.0 # seconds, 0.0 = forever
CAMERA_FRAME    = "base_front"


# ───────────────────────── NODE CLASS ─────────────────────────────
class SemanticMapper:
    def __init__(self):
        rospy.init_node("semantic_mapper")

        # --- PUB/SUB -------------------------------------------------------
        self.pub_markers = rospy.Publisher(
            "semantic_map_markers", MarkerArray, queue_size=10, latch=True)

        rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes,
                         self.cb_boxes, queue_size=5)
        rospy.Subscriber("/scan", LaserScan, self.cb_scan, queue_size=5)
        rospy.Subscriber("/map",  OccupancyGrid, self.cb_map,  queue_size=1)
        rospy.Subscriber("/head/camera/camera_info", CameraInfo,
                         self.cb_caminfo, queue_size=1)

        # --- TF ------------------------------------------------------------
        self.tf_buf = tf2_ros.Buffer(rospy.Duration(30))
        tf2_ros.TransformListener(self.tf_buf)

        # --- STATE ---------------------------------------------------------
        self.latest_scan = None
        self.map_frame   = "map"
        self.marker_id   = itertools.count()
        self.img_w, self.hfov = 640, math.radians(90)

        # id → {class_, pos, confirmed, weight}
        self.records: Dict[int, dict] = {}
        self.cluster_persist: Dict[frozenset, int] = {}

        rospy.Timer(rospy.Duration(CLUSTER_PERIOD), self.timer_cluster)
        rospy.loginfo("semantic_mapper ready")


    # ─────────────── CALLBACKS ──────────────────────────────────────
    def cb_caminfo(self, msg: CameraInfo):
        self.img_w = msg.width
        fx         = msg.K[0]
        if fx > 0:
            self.hfov = 2.0 * math.atan2(0.5 * self.img_w, fx)


    def cb_map(self, msg: OccupancyGrid):
        self.map_frame = msg.header.frame_id or "map"


    def cb_scan(self, msg: LaserScan):
        self.latest_scan = msg


    # --------------- Bounding boxes ---------------------------------------
    def cb_boxes(self, msg: BoundingBoxes):
        if self.latest_scan is None:
            return

        a_min, a_inc = self.latest_scan.angle_min, self.latest_scan.angle_increment
        ranges       = self.latest_scan.ranges
        N_beams      = len(ranges)

        m_arr = MarkerArray()
        stamp = rospy.Time.now()

        for box in msg.bounding_boxes:
            cls = box.Class.lower()
            if cls not in SUPPORTED_CLASSES:
                continue

            # check image center
            u_mid = 0.5 * (box.xmin + box.xmax)
            if abs(u_mid - self.img_w/2) > CENTER_TOLERANCE * (self.img_w/2):
                continue

            # ray angle
            theta = ((u_mid - self.img_w/2) / self.img_w) * self.hfov
            idx   = int(round((theta - a_min) / a_inc))
            if not (0 <= idx < N_beams):
                continue

            rng = ranges[idx]
            if not (math.isfinite(rng) and 0.0 < rng <= MAX_MAPPING_RANGE):
                continue

            # point in camera frame
            pt_cam = PointStamped()
            pt_cam.header.stamp    = self.latest_scan.header.stamp
            pt_cam.header.frame_id = CAMERA_FRAME
            pt_cam.point.x = rng * math.cos(theta)
            pt_cam.point.y = rng * math.sin(theta)
            pt_cam.point.z = 0.0

            # transform into the map
            try:
                pt_map = self.tf_buf.transform(pt_cam, self.map_frame,
                                               rospy.Duration(0.2))
            except tf2_ros.TransformException:
                continue

            # сreate a marker
            m = self.make_marker(cls, pt_map.point, stamp)
            m_arr.markers.append(m)

            # save record
            self.records[m.id] = dict(class_=cls,
                                      pos=np.array([pt_map.point.x,
                                                    pt_map.point.y,
                                                    pt_map.point.z]),
                                      confirmed=False,
                                      weight=1)

        if m_arr.markers:
            self.pub_markers.publish(m_arr)


    # ─────────────── CLUSTER TIMER ─────────────────────────────────
    def timer_cluster(self, _):
        ids = list(self.records.keys())
        if not ids:
            self.cluster_persist.clear()
            return

        positions = np.array([self.records[i]["pos"] for i in ids])
        unvis = set(range(len(ids)))
        clusters: List[Set[int]] = []

        while unvis:
            root = unvis.pop()
            comp = {root}; stack=[root]
            while stack:
                i = stack.pop()
                d = np.linalg.norm(positions[list(unvis)] - positions[i], axis=1)
                neigh = [j for j, dist in zip(list(unvis), d) if dist <= CLUSTER_DIST]
                for j in neigh:
                    unvis.remove(j)
                    stack.append(j)
                    comp.add(j)
            clusters.append({ids[k] for k in comp})

        new_persist = {}
        for cl in clusters:
            if len(cl) > 1:
                key = frozenset(cl)
                cnt = self.cluster_persist.get(key, 0) + 1
                if cnt >= CLUSTER_CONFIRM_ITERS:
                    self.collapse_cluster(cl)
                else:
                    new_persist[key] = cnt
        self.cluster_persist = new_persist

        # delete confirmed records with single detections
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
                    del self.records[mid]
        if del_arr.markers:
            self.pub_markers.publish(del_arr)


    # ─────────────── CLUSTER COLLAPSE ─────────────────────────────
    def collapse_cluster(self, ids: Set[int]):
        classes   = [self.records[i]["class_"] for i in ids]
        positions = np.array([self.records[i]["pos"]   for i in ids])
        weights   = np.array([self.records[i]["weight"] for i in ids])

        centroid  = np.average(positions, axis=0, weights=weights)

        # ---------- class picking ----------
        votes: Dict[str, float] = {}
        for cls, w in zip(classes, weights):
            votes[cls] = votes.get(cls, 0.0) + w

        majority = max(votes, key=votes.get)
        # sofa has priority over chair
        if "sofa" in votes and "chair" in votes:
            majority = "sofa"

        # ---------- delete old markers ---------------------------------
        del_arr = MarkerArray()
        for mid in ids:
            rec = self.records[mid]
            m_del = Marker()
            m_del.header.frame_id = self.map_frame
            m_del.header.stamp    = rospy.Time.now()
            m_del.ns   = rec["class_"]
            m_del.id   = mid
            m_del.action = Marker.DELETE
            del_arr.markers.append(m_del)
            del self.records[mid]
        if del_arr.markers:
            self.pub_markers.publish(del_arr)

        # ---------- out of cluster made marker -------------------------
        pt  = PointStamped()
        pt.point.x, pt.point.y, pt.point.z = centroid
        new_m = self.make_marker(majority, pt.point, rospy.Time.now())
        self.pub_markers.publish(MarkerArray([new_m]))

        # ---------- save new record -----------------------------------
        self.records[new_m.id] = dict(class_=majority,
                                      pos=centroid,
                                      confirmed=True,
                                      weight=1)


    # ─────────────── MARKER FACTORY ───────────────────────────────
    def make_marker(self, cls_name, pt_map, stamp):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = stamp
        m.ns   = cls_name
        m.id   = next(self.marker_id)
        m.type, m.action = Marker.CUBE, Marker.ADD

        m.pose.position.x, m.pose.position.y, m.pose.position.z = \
            pt_map.x, pt_map.y, pt_map.z
        m.pose.orientation.w = 1.0

        r, g, b, a = COLORS.get(cls_name, (1, 1, 0, 0.9))
        m.color = ColorRGBA(r, g, b, a)
        s = SIZE_M.get(cls_name, 0.4)
        m.scale.x = m.scale.y = m.scale.z = s
        m.lifetime = rospy.Duration(MARKER_LIFETIME)
        return m


# ───────────────────────── MAIN ───────────────────────────────────
if __name__ == "__main__":
    try:
        SemanticMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
