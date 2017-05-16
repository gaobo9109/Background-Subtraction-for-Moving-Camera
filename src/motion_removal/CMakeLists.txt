file(GLOB motion_removal_SRC *.cpp *.h)

add_library(motion_removal ${motion_removal_SRC})
target_link_libraries(motion_removal ${OpenCV_LIBRARIES} ${GSL_LIBRARIES} ${util})



