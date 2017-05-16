include_directories("include")
file(GLOB util_SRC src/*.cpp)

add_library(util ${util_SRC})
target_link_libraries(util ${OpenCV_LIBRARIES})

