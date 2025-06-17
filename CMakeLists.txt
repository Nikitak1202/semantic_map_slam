cmake_minimum_required(VERSION 3.0.2)
project(semantic_map_slam)

##############################################################################
# 1. Зависимости пакета
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
# 2. Сообщения
##############################################################################
add_message_files(
  FILES
    SemanticGrid.msg          # описание семантического грида
)

generate_messages(
  DEPENDENCIES
  std_msgs nav_msgs
  std_msgs                 # timestamp / header
  nav_msgs                 # OccupancyGrid
  sensor_msgs              # Image  ← добавили
)


##############################################################################
# 3. Экспорт для других пакетов
##############################################################################
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    nav_msgs
    vision_msgs
    tf2_ros
    message_runtime          # runtime-часть для сгенерированных сообщений
)

##############################################################################
# 4. Сборка C++-плагинов для Gazebo (mecanum)
##############################################################################
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(mecanum SHARED src/mecanum_controller.cc)
target_link_libraries(mecanum ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES})

##############################################################################
# 5. Установка Python-скриптов
##############################################################################
catkin_install_python(
  PROGRAMS
    src/semantic_mapper.py    # узел, который красит карту по YOLO
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

##############################################################################
# 6. Установка launch- и world-файлов
##############################################################################
install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY worlds
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
