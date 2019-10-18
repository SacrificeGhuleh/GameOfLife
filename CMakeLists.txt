cmake_minimum_required(VERSION 3.14)
project(GameOfLife)
set(CMAKE_CXX_STANDARD 14)

# Where to find CMake modules and OpenCV
SET(OpenCV_DIR "C:/opencv-4.1.1/mingw-build/install_release")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
INCLUDE(MacroOpenCV)
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp timer.h)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)