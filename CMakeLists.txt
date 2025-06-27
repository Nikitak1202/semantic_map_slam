cmake_minimum_required(VERSION 3.0.2)
project(semantic_map_slam)

##############################################################################
# 1. Package dependencies
##############################################################################
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs
  vision_msgs
  sensor_msgs
  image_transport
  cv_bridge
  tf2_ros
  message_generation   # для msg-файла
)

##############################################################################
# 2. Messages
##############################################################################
add_message_files(FILES)

generate_messages(
  DEPENDENCIES
  std_msgs nav_msgs
  std_msgs                 # timestamp / header
  nav_msgs                 # OccupancyGrid
  sensor_msgs              # Image  
)


##############################################################################
# 3. Exporting package information
##############################################################################
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    nav_msgs
    vision_msgs
    tf2_ros
    message_runtime          
)

##############################################################################
# 4. Building C++ plugin for Gazebo (mecanum)
##############################################################################
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(mecanum SHARED src/mecanum_controller.cc)
target_link_libraries(mecanum ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES})

##############################################################################
# 5. Python scripts
##############################################################################
catkin_install_python(
  PROGRAMS
    src/semantic_mapper.py   
    src/explore_scene.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

##############################################################################
# 6. Install launch files and worlds
##############################################################################
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY worlds
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
