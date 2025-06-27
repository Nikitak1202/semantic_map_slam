# Semantic-Map-SLAM Demo Scene  

## 1. What is it?
This repo is a **Gazebo + ROS** scene with a small service robot.  
The robot can:

* drive with 4 mecanum wheels (holonomic base);
* build a 2-D map with **Hector SLAM**;
* find objects with **YOLO v2-tiny** and add them as colored cubes;
* plan a path (A*). If no path exists, it rolls to the nearest border;
* visit every zone of the room.

Everything runs on **ROS Noetic** (Ubuntu 20.04) and **Gazebo 11**.

---

## 2. Demo  

<video src="demo.mp4" controls width="720"></video>  

The video will start when you click ▶.

---

## 3. How to install

> Tested on Ubuntu 20.04 + ROS Noetic.

### 3.1 System packages  

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full \
                 ros-noetic-gazebo-ros-pkgs \
                 ros-noetic-gazebo-ros-control \
                 ros-noetic-hector-mapping \
                 ros-noetic-tf2-ros \
                 ros-noetic-vision-msgs \
                 python3-catkin-tools \
                 python3-rosdep git
sudo rosdep init
rosdep update
