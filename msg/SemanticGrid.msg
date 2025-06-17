# === Header + original occupancy grid (0–100, -1 = unknown) =========
std_msgs/Header         header
nav_msgs/OccupancyGrid  grid

# === For every unique cell value, which class-id it represents ======
uint8[]                 class_ids   # same length as data[] in grid

# === A colourised preview   ===============================
sensor_msgs/Image       preview