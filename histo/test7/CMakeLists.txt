cmake_minimum_required(VERSION 2.8)
project(calcHist_Demo)
find_package(OpenCV REQUIRED)
add_executable(calcHist_Demo calcHist_Demo.cpp)
target_link_libraries(calcHist_Demo ${OpenCV_LIBS})

