CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)
project(bgs)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
FIND_PACKAGE(GSL REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIRS}
  ${GSL_INCLUDE_DIRS}
)


#file(GLOB sgm src/sgm/*.cpp src/sgm/*.h)
#file(GLOB sgm_grid src/sgm_grid/*.cpp src/sgm_grid/*.h)
file(GLOB util src/util/*.cpp src/util/*.h)
file(GLOB gmm src/gmm/*.cpp src/gmm/*.h)
file(GLOB motion_removal src/rgbd_motion_removal/*.cpp src/rgbd_motion_removal/*.h)

add_library(${PROJECT_NAME}
#    ${sgm}
    ${util}
#    ${sgm_grid}
    ${gmm}
    ${motion_removal}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    ${GSL_LIBRARIES}
)

add_executable(bgs_bin src/main.cpp)

target_link_libraries(bgs_bin
    ${PROJECT_NAME}
)



