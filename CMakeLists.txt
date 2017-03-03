CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
project(dual_sgm)

find_package(OpenCV REQUIRED) # core imgproc highgui flann video calib3d nonfree objdetect)

add_executable(${PROJECT_NAME} src/main.cpp src/dual_sgm.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})



