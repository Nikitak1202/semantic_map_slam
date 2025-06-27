#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zone-exploration FSM with A* global planner + local fallback.
If A* fails for a zone, robot drives to its border (local control),
marks the zone reached, and continues as usual.
RViz topics:
  • /explore_zones  – MarkerArray   (all zones, green/grey)
  • /plan_path      – nav_msgs/Path (A* result)
  • /plan_wp        – Marker        (current waypoint)
"""
import math, heapq, numpy as np, rospy, tf
from geometry_msgs.msg     import Twist, PoseStamped
from nav_msgs.msg          import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray


# ─────────── CONFIG ─────────────────────────────────────────────────
ZONES_X, ZONES_Y = 5, 5
ROBOT_RADIUS     = 0.5    # m
SAFETY_GAP       = 0.1    # m
MAX_SPEED        = 1      # m/s
LOOK_AHEAD       = 0.3    # m for pure pursuit
SPIN_VEL         = 0.3    # rad/s
GOAL_TOL         = 0.5    # m
PRINT_RATE       = 20     # Hz


# ─────────── UTILITIES ─────────────────────────────────────────────
def angle_diff(a, b):
    d = a - b
    return (d + math.pi) % (2*math.pi) - math.pi


def astar(grid, start, goal):
    """A* on 4-neighbour boolean grid (True = blocked)."""
    h, w        = grid.shape
    sx, sy      = start
    gx, gy      = goal
    if grid[sy, sx] or grid[gy, gx]:
        return None
    open_set     = [(abs(gx-sx)+abs(gy-sy), 0, sx, sy)]
    came, gscore = {}, {(sx, sy): 0}
    while open_set:
        _, g, x, y = heapq.heappop(open_set)
        if (x, y) == (gx, gy):
            break
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= nx < w and 0 <= ny < h and not grid[ny, nx]:
                ng = g + 1
                if ng < gscore.get((nx, ny), 1e9):
                    gscore[(nx, ny)] = ng
                    f = ng + abs(gx-nx) + abs(gy-ny)
                    heapq.heappush(open_set, (f, ng, nx, ny))
                    came[(nx, ny)] = (x, y)
    else:
        return None
    # reconstruct
    path = [(gx, gy)]
    while path[-1] != (sx, sy):
        path.append(came[path[-1]])
    path.reverse()
    return path


# ─────────── NODE ─────────────────────────────────────────────────
class ExploreScene:
    IDLE, SELECT, PLAN, FOLLOW, SPIN, DONE = range(6)
    NAME = {
        IDLE:   'IDLE',
        SELECT: 'SELECT',
        PLAN:   'PLAN',
        FOLLOW: 'FOLLOW',
        SPIN:   'SPIN',
        DONE:   'DONE',
    }


    def __init__(self):
        rospy.init_node('explore_scene')
        self.state       = self.IDLE
        self.goal        = None
        self.map_ok      = False
        self.visited     = None
        self.pose        = (0.0, 0.0, 0.0)  # x,y,yaw
        self.path_pts    = []
        self.cur_wp_idx  = 0

        # fallback to border
        self.fallback       = False
        self.local_target   = None

        # publishers / subscribers
        self.cmd_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.zone_pub = rospy.Publisher('/explore_zones', MarkerArray,
                                        queue_size=1, latch=True)
        self.path_pub = rospy.Publisher('/plan_path', Path,
                                        queue_size=1, latch=True)
        self.wp_pub   = rospy.Publisher('/plan_wp', Marker,
                                        queue_size=1, latch=True)

        rospy.Subscriber('/map', OccupancyGrid, self.map_cb, queue_size=1)

        # no need to subscribe scan for fallback
        self.tfl = tf.TransformListener()
        rospy.Timer(rospy.Duration(1.0/PRINT_RATE), self.loop)


    # ── map callback ───────────────────────────────────────────
    def map_cb(self, msg):
        info          = msg.info
        self.res      = info.resolution
        self.min_x    = info.origin.position.x
        self.min_y    = info.origin.position.y
        self.w, self.h= info.width, info.height

        # initialize visited mask once
        if self.visited is None:
            self.zone_w = (self.w * self.res) / ZONES_X
            self.zone_h = (self.h * self.res) / ZONES_Y
            self.visited = np.zeros((ZONES_X, ZONES_Y), dtype=bool)

        # occupancy grid -> boolean blocked map
        occ = np.array(msg.data, dtype=np.int8).reshape(self.h, self.w) > 50
        # inflate obstacles by robot radius + gap
        rad = int(round((ROBOT_RADIUS + SAFETY_GAP) / self.res))
        if rad > 0:
            inflated = np.zeros_like(occ)
            for dx in range(-rad, rad+1):
                for dy in range(-rad, rad+1):
                    inflated |= np.roll(np.roll(occ, dx, axis=1), dy, axis=0)
            occ = inflated
        self.block = occ
        self.map_ok = True
        self.pub_zones()


    # ── TF pose ───────────────────────────────────────────────
    def tf_pose(self):
        try:
            t, r = self.tfl.lookupTransform('map', 'base_front', rospy.Time(0))
            yaw   = tf.transformations.euler_from_quaternion(r)[2]
            self.pose = (t[0], t[1], yaw)
            return True
        except (tf.Exception, tf.ConnectivityException):
            return False


    # ── zone helpers ───────────────────────────────────────────
    def zone_of(self, x, y):
        ix = int(np.clip((x - self.min_x) / self.zone_w, 0, ZONES_X-1))
        iy = int(np.clip((y - self.min_y) / self.zone_h, 0, ZONES_Y-1))
        return ix, iy


    def zone_center(self, idx):
        return (
            self.min_x + (idx[0] + 0.5) * self.zone_w,
            self.min_y + (idx[1] + 0.5) * self.zone_h
        )


    def zone_free(self, idx):
        ix, iy = idx
        x0 = int((ix * self.zone_w) / self.res)
        x1 = int(((ix+1) * self.zone_w) / self.res)
        y0 = int((iy * self.zone_h) / self.res)
        y1 = int(((iy+1) * self.zone_h) / self.res)
        x0, x1 = np.clip([x0, x1], 0, self.w-1)
        y0, y1 = np.clip([y0, y1], 0, self.h-1)
        sub = self.block[y0:y1+1, x0:x1+1]
        return np.any(~sub)


    # ── publish zone markers ───────────────────────────────────
    def pub_zones(self):
        ma    = MarkerArray()
        stamp = rospy.Time.now()
        for ix in range(ZONES_X):
            for iy in range(ZONES_Y):
                m = Marker()
                m.header.frame_id, m.header.stamp = 'map', stamp
                m.ns, m.id     = 'zones', ix*ZONES_Y + iy
                m.type, m.action = Marker.CUBE, Marker.ADD
                cx, cy         = self.zone_center((ix, iy))
                m.pose.position.x, m.pose.position.y = cx, cy
                m.pose.position.z = 0.02
                m.pose.orientation.w = 1.0
                m.scale.x, m.scale.y, m.scale.z = self.zone_w, self.zone_h, 0.02
                if self.visited[ix, iy]:
                    m.color.r, m.color.g, m.color.b = 0.6, 0.6, 0.6
                else:
                    m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
                m.color.a = 0.3
                ma.markers.append(m)
        self.zone_pub.publish(ma)


    # ── publish global path ─────────────────────────────────────
    def pub_path(self):
        if not self.path_pts:
            return
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp    = rospy.Time.now()
        for mx, my in self.path_pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = self.min_x + mx * self.res
            ps.pose.position.y = self.min_y + my * self.res
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)


    # ── publish waypoint marker ───────────────────────────────────
    def pub_wp(self):
        if self.fallback:
            tx, ty = self.local_target
        else:
            if self.cur_wp_idx >= len(self.path_pts):
                return
            mx, my = self.path_pts[self.cur_wp_idx]
            tx = self.min_x + mx * self.res
            ty = self.min_y + my * self.res

        m = Marker()
        m.header.frame_id, m.header.stamp = 'map', rospy.Time.now()
        m.ns, m.id     = 'wp', 0
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y = tx, ty
        m.pose.position.z = 0.05
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b, m.color.a = 1, 0, 1, 0.9
        self.wp_pub.publish(m)


    # ── command helper ───────────────────────────────────────────
    def publish_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = vx, vy, wz
        self.cmd_pub.publish(cmd)


    # ── main FSM loop ────────────────────────────────────────────
    def loop(self, _):
        if not (self.map_ok and self.tf_pose()):
            rospy.loginfo_throttle(1, 'Waiting for map/TF…')
            return

        x, y, yaw = self.pose

        if self.state == self.IDLE:
            self.state = self.SELECT

        # ─── SELECT ─────────────────────────────────────────────
        if self.state == self.SELECT:
            self.goal      = None
            self.fallback  = False
            cur = self.zone_of(x, y)
            if not self.visited[cur]:
                if self.zone_free(cur):
                    self.goal = cur
                else:
                    self.visited[cur] = True
                    self.pub_zones()
            if self.goal is None:
                unvis = list(zip(*np.where(~self.visited)))
                if not unvis:
                    self.state = self.DONE
                    self.publish_cmd()
                    return
                centers = np.array([self.zone_center(z) for z in unvis])
                dists   = np.hypot(centers[:,0]-x, centers[:,1]-y)
                self.goal = tuple(unvis[int(dists.argmin())])
            self.state = self.PLAN

        # ─── PLAN ───────────────────────────────────────────────
        elif self.state == self.PLAN:
            sx = int((x - self.min_x)/self.res)
            sy = int((y - self.min_y)/self.res)
            cx, cy = self.zone_center(self.goal)
            gx = int((cx - self.min_x)/self.res)
            gy = int((cy - self.min_y)/self.res)
            path = astar(self.block, (sx, sy), (gx, gy))

            if path:
                # global-follow
                self.path_pts   = path
                self.cur_wp_idx = 1
                self.fallback   = False
                self.pub_path()
            else:
                # fallback: drive to border of zone
                self.fallback = True
                # compute intersection with zone rectangle
                zx, zy = cx, cy
                dx, dy = zx - x, zy - y
                # parametric t until hitting boundary of zone
                half_w, half_h = self.zone_w/2, self.zone_h/2
                # compute border in world coords
                t_vals = []
                if dx != 0:
                    tx1 = ((self.min_x + self.goal[0]*self.zone_w) - x)/dx
                    tx2 = ((self.min_x + (self.goal[0]+1)*self.zone_w) - x)/dx
                    t_vals += [tx1, tx2]
                if dy != 0:
                    ty1 = ((self.min_y + self.goal[1]*self.zone_h) - y)/dy
                    ty2 = ((self.min_y + (self.goal[1]+1)*self.zone_h) - y)/dy
                    t_vals += [ty1, ty2]
                # select smallest positive t
                t = min([tv for tv in t_vals if tv>0], default=1.0)
                tx = x + dx * min(t, 1.0)
                ty = y + dy * min(t, 1.0)
                self.local_target = (tx, ty)

            self.pub_wp()
            self.state = self.FOLLOW

        # ─── FOLLOW ─────────────────────────────────────────────
        elif self.state == self.FOLLOW:
            if self.fallback:
                tx, ty = self.local_target
                dist = math.hypot(tx - x, ty - y)
                # drive directly toward border
                if dist < GOAL_TOL:
                    self.visited[self.goal] = True
                    self.pub_zones()
                    self.publish_cmd()
                    self.state = self.SELECT
                else:
                    yaw_err = math.atan2(ty-y, tx-x) - yaw
                    # holonomic base
                    vx = MAX_SPEED * math.cos(yaw_err)
                    vy = MAX_SPEED * math.sin(yaw_err)
                    self.publish_cmd(vx, vy, 0.0)
            else:
                # pure-pursuit along path_pts
                if self.cur_wp_idx >= len(self.path_pts):
                    self.state      = self.SPIN
                    self.spin_accum = 0
                    self.prev_yaw   = yaw
                else:
                    mx, my = self.path_pts[self.cur_wp_idx]
                    wx     = self.min_x + mx * self.res
                    wy     = self.min_y + my * self.res
                    if math.hypot(wx-x, wy-y) < LOOK_AHEAD:
                        self.cur_wp_idx += 1
                        self.pub_wp()
                    else:
                        dx, dy = wx - x, wy - y
                        vx =  dx*math.cos(yaw) + dy*math.sin(yaw)
                        vy = -dx*math.sin(yaw) + dy*math.cos(yaw)
                        norm = math.hypot(vx, vy)
                        if norm > MAX_SPEED:
                            vx, vy = vx*MAX_SPEED/norm, vy*MAX_SPEED/norm
                        self.publish_cmd(vx, vy, 0.0)

        # ─── SPIN ───────────────────────────────────────────────
        elif self.state == self.SPIN:
            self.publish_cmd(0, 0, SPIN_VEL)
            self.spin_accum += abs(angle_diff(self.pose[2], self.prev_yaw))
            self.prev_yaw   = self.pose[2]
            if self.spin_accum >= 2*math.pi - 0.1:
                self.visited[self.goal] = True
                self.pub_zones()
                self.state = self.SELECT

        # ─── LOGGING ───────────────────────────────────────────
        rospy.loginfo_throttle(
            1/PRINT_RATE,
            f"STATE={self.NAME[self.state]}  "
            f"visited={int(self.visited.sum())}/{ZONES_X*ZONES_Y}  "
            f"goal={self.goal}"
        )


# ─────────── MAIN ─────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        ExploreScene()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
