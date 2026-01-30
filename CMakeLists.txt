cmake_minimum_required(VERSION 3.16)
project(zifra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(zifra src/main.cpp src/stitcher.cpp)

find_package(OpenCV REQUIRED)
target_include_directories(zifra PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(zifra PRIVATE ${OpenCV_LIBS})
