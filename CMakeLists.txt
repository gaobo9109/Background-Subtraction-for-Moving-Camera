CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)
project(bgs)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)

include_directories(
  src/util/include
  ${EIGEN3_INCLUDE_DIRS}
  ${GSL_INCLUDE_DIRS}
)

add_subdirectory(src/util)
add_subdirectory(src/gmm)
add_subdirectory(src/motion_removal)
add_subdirectory(src/dvo_integration)

add_executable(bgs_bin src/main.cpp)

target_link_libraries(bgs_bin
    util
    gmm
    motion_removal
)



