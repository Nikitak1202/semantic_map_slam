# Semantic Map SLAM Project

## What is it?

This project simulates a mobile robot equipped with a Mecanum drive, RGB camera, and lidar. The robot builds a semantic map using YOLO object detection, SLAM (Hector Mapping), and navigates autonomously within a room environment using global path planning (A\*) and a local fallback planner.

Key features:

* **ROS-based robot simulation** in Gazebo
* **Semantic mapping**: object detection with YOLO
* **Navigation**: Autonomous zone exploration using A\* and local fallback

## Demo

Watch the demo video [here](./demo.mp4).

## How to install

### Step 1: Dependencies

Install ROS Noetic and Gazebo 11:

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full ros-noetic-gazebo-ros-control ros-noetic-hector-slam ros-noetic-darknet-ros ros-noetic-controller-manager
```

### Step 2: Prepare workspace

Create and build the ROS workspace:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone <this-repo-url>
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

### Step 3: Run simulation

Start simulation with RViz and Gazebo:

```bash
roslaunch semantic_map_slam main.launch
```

Your environment is ready! Now the robot will autonomously explore the room, build a semantic map, and classify detected objects.
