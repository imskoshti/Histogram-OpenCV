cmake_minimum_required(VERSION 2.8)
project(calcHist)
find_package(OpenCV REQUIRED)
add_executable(calcHist calcHist.cpp)
target_link_libraries(calcHist ${OpenCV_LIBS})


