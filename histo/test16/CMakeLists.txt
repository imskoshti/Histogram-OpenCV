cmake_minimum_required(VERSION 2.8)
project(1d_histogram_on_opencv)
find_package(OpenCV REQUIRED)
add_executable(1d_histogram_on_opencv 1d_histogram_on_opencv.cpp)
target_link_libraries(1d_histogram_on_opencv ${OpenCV_LIBS})

