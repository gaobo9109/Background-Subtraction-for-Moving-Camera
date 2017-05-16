file(GLOB gmm_SRC *.cpp *.h)

add_library(gmm ${gmm_SRC})
target_link_libraries(gmm ${OpenCV_LIBRARIES} util)
