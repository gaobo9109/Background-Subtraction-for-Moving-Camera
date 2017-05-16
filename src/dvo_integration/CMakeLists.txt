find_package(catkin REQUIRED COMPONENTS roscpp dvo_core dvo_ros cv_bridge image_transport)

include_directories(
    ${dvo_core_INCLUDE_DIRS}
    ${dvo_ros_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

add_executable(bgs_ros dense_tracker_bgs.cpp)

target_link_libraries(bgs_ros
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    util
    gmm
)

add_executable(dvo_test dvo_test.cpp)
target_link_libraries(dvo_test
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    util
)



