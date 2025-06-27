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



https://github.com/user-attachments/assets/5d64fcbe-e0e2-4c32-a0f7-a742b85c9154



---

## 3. How to run

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
```

### 3.2 Create a catkin workspace and clone repos
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/leggedrobotics/darknet_ros.git
git clone https://github.com/Nikitak1202/semantic_map_slam.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin build
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3.3 Run the simulation
```bash
roslaunch semantic_map_slam slam.launch gazebo_gui:=false rviz_gui:=true
```
