#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous exploration FSM with zone / goal visualisation for RViz.
"""

# ── CONFIG ────────────────────────────────────────────────────────────────
ZONES_X, ZONES_Y = 4, 4      # grid resolution
LIN_VEL          = 1       # [m/s]
ANG_VEL          = 1.0       # [rad/s] for avoidance
SPIN_VEL         = 1.0       # [rad/s] full spin
GOAL_TOL         = 2      # [m]
SAFE_DIST        = 1       # [m] lidar front arc
PRINT_RATE       = 20        # [Hz]

# ── ROS / STD ─────────────────────────────────────────────────────────────
import math, numpy as np, rospy, tf
from geometry_msgs.msg import Twist
from nav_msgs.msg      import Odometry, OccupancyGrid
from sensor_msgs.msg   import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

# ── FSM IMPLEMENTATION ───────────────────────────────────────────────────
class ExploreScene:
    IDLE, SELECT, DRIVE, AVOID, SPIN, DONE = range(6)
    STATE_NAMES = {IDLE:'IDLE', SELECT:'SELECT', DRIVE:'DRIVE',
                   AVOID:'AVOID', SPIN:'SPIN', DONE:'DONE'}

    def __init__(self):
        rospy.init_node('explore_scene')
        self.state       = self.IDLE
        self.map_ready   = False
        self.pose        = (0.0, 0.0, 0.0)
        self.scan_ranges = []
        self.spin_accum  = 0.0
        self.prev_yaw    = 0.0
        self.goal        = None
        self.zone_w = self.zone_h = 1.0   # dummy until map arrives

        self.cmd_pub   = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.zone_pub  = rospy.Publisher('/explore_zones', MarkerArray,
                                         queue_size=1, latch=True)
        self.goal_pub  = rospy.Publisher('/explore_goal', Marker,
                                         queue_size=1, latch=True)

        rospy.Subscriber('/odom', Odometry,      self.odom_cb,  queue_size=1)
        rospy.Subscriber('/scan', LaserScan,     self.scan_cb,  queue_size=1)
        rospy.Subscriber('/map',  OccupancyGrid, self.map_cb,   queue_size=1)

        rospy.Timer(rospy.Duration(1.0/PRINT_RATE), self.loop)

    # ── Callbacks ────────────────────────────────────────────────────────
    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        self.pose = (msg.pose.pose.position.x,
                     msg.pose.pose.position.y,
                     yaw)

    def scan_cb(self, msg):
        self.scan_ranges = msg.ranges

    def map_cb(self, msg: OccupancyGrid):
        info = msg.info
        self.min_x = info.origin.position.x
        self.min_y = info.origin.position.y
        self.max_x = self.min_x + info.width  * info.resolution
        self.max_y = self.min_y + info.height * info.resolution

        self.zone_w = (self.max_x - self.min_x) / ZONES_X
        self.zone_h = (self.max_y - self.min_y) / ZONES_Y

        self.occ = np.array(msg.data, dtype=np.int8).reshape(info.height,
                                                             info.width)
        if not hasattr(self, 'visited'):
            self.visited = np.zeros((ZONES_X, ZONES_Y), dtype=bool)
        self.map_ready = True
        self.publish_zone_markers()          # update grid in RViz

    # ── Helpers ──────────────────────────────────────────────────────────
    def zone_of(self, x, y):
        ix = int(np.clip((x - self.min_x) / self.zone_w, 0, ZONES_X-1))
        iy = int(np.clip((y - self.min_y) / self.zone_h, 0, ZONES_Y-1))
        return ix, iy

    def zone_center(self, idx):
        cx = self.min_x + (idx[0] + 0.5)*self.zone_w
        cy = self.min_y + (idx[1] + 0.5)*self.zone_h
        return cx, cy

    def front_blocked(self):
        if not self.scan_ranges: return False
        half = len(self.scan_ranges)//2
        arc  = self.scan_ranges[half-30:half+30]
        return min(arc) < SAFE_DIST

    def angle_diff(self, a, b):
        d = a - b
        return (d + math.pi) % (2*math.pi) - math.pi

    # ── Motion primitives ────────────────────────────────────────────────
    def publish_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        cmd = Twist()
        cmd.linear.x  = vx
        cmd.linear.y  = vy
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

    def move_to(self, gx, gy):
        x, y, yaw = self.pose
        dx, dy = gx - x, gy - y
        tgt_yaw = math.atan2(dy, dx)
        yaw_err = self.angle_diff(tgt_yaw, yaw)
        vx = LIN_VEL * math.cos(yaw_err)
        vy = LIN_VEL * math.sin(yaw_err)
        self.publish_cmd(vx, vy, 0.5*yaw_err)

    def turn_in_place(self, speed):
        self.publish_cmd(0.0, 0.0, speed)

    def reached_goal(self, gx, gy):
        x, y, _ = self.pose
        return math.hypot(gx - x, gy - y) < GOAL_TOL

    def zone_free(self, idx):
        cx, cy = self.zone_center(idx)
        mx = int((cx - self.min_x) / (self.max_x - self.min_x) *
                 (self.occ.shape[1]-1))
        my = int((cy - self.min_y) / (self.max_y - self.min_y) *
                 (self.occ.shape[0]-1))
        return self.occ[my, mx] <= 0

    # ── Visualization ───────────────────────────────────────────────────
    def publish_zone_markers(self):
        ma = MarkerArray()
        stamp = rospy.Time.now()
        for ix in range(ZONES_X):
            for iy in range(ZONES_Y):
                m = Marker()
                m.header.frame_id = 'map'
                m.header.stamp    = stamp
                m.ns  = 'zones'
                m.id  = ix*ZONES_Y + iy
                m.type= Marker.CUBE
                m.action = Marker.ADD
                cx, cy = self.zone_center((ix, iy))
                m.pose.position.x = cx
                m.pose.position.y = cy
                m.pose.position.z = 0.01
                m.pose.orientation.w = 1.0
                m.scale.x = self.zone_w
                m.scale.y = self.zone_h
                m.scale.z = 0.02
                if self.visited[ix, iy]:
                    m.color.r, m.color.g, m.color.b = 0.6, 0.6, 0.6  # grey
                else:
                    m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0  # green
                m.color.a = 0.3
                ma.markers.append(m)
        self.zone_pub.publish(ma)

    def publish_goal_marker(self):
        if self.goal is None: return
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = rospy.Time.now()
        m.ns  = 'goal'
        m.id  = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        cx, cy = self.zone_center(self.goal)
        m.pose.position.x = cx
        m.pose.position.y = cy
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = self.zone_w * 0.7
        m.scale.y = self.zone_h * 0.7
        m.scale.z = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.8
        self.goal_pub.publish(m)

    # ── Main loop ────────────────────────────────────────────────────────
    def loop(self, _):
        if not self.map_ready:
            rospy.loginfo_throttle(1.0, 'Waiting for /map…')
            return

        if self.state == self.IDLE:
            self.state = self.SELECT

        elif self.state == self.SELECT:
            x, y, _ = self.pose
            current = self.zone_of(x, y)
            if not self.visited[current] and self.zone_free(current):
                self.goal = current
            else:
                unvis = [(ix, iy) for ix in range(ZONES_X)
                                   for iy in range(ZONES_Y)
                                   if not self.visited[ix, iy]
                                   and self.zone_free((ix, iy))]
                if not unvis:
                    self.state = self.DONE
                    self.goal  = None
                    self.publish_cmd()
                    self.publish_goal_marker()
                    self.publish_zone_markers()
                    return
                gx, gy = zip(*[self.zone_center(idx) for idx in unvis])
                dist = np.hypot(np.array(gx)-x, np.array(gy)-y)
                self.goal = unvis[int(dist.argmin())]
            self.publish_goal_marker()
            self.state = self.DRIVE

        elif self.state == self.DRIVE:
            if self.front_blocked():
                self.state = self.AVOID
            else:
                gx, gy = self.zone_center(self.goal)
                self.move_to(gx, gy)
                if self.reached_goal(gx, gy):
                    self.state      = self.SPIN
                    self.spin_accum = 0.0
                    self.prev_yaw   = self.pose[2]

        elif self.state == self.AVOID:
            if self.front_blocked():
                self.turn_in_place(ANG_VEL)
            else:
                self.state = self.DRIVE

        elif self.state == self.SPIN:
            self.turn_in_place(SPIN_VEL)
            yaw = self.pose[2]
            d   = abs(self.angle_diff(yaw, self.prev_yaw))
            self.spin_accum += d
            self.prev_yaw = yaw
            if self.spin_accum >= 2*math.pi - 0.1:
                self.visited[self.goal] = True
                self.publish_zone_markers()
                self.publish_cmd()
                self.state = self.SELECT

        # ── console status ─────────────────────────────────────────────
        visited_cnt = int(self.visited.sum())
        goal_str = 'None' if self.goal is None else str(self.goal)
        state_str = self.STATE_NAMES[self.state]
        rospy.loginfo_throttle(1.0/PRINT_RATE,
            f'STATE={state_str} visited={visited_cnt}/{ZONES_X*ZONES_Y} goal={goal_str}')

        # update goal marker continuously
        self.publish_goal_marker()

# ── main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        ExploreScene()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
