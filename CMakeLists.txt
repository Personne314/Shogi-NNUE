cmake_minimum_required(VERSION 3.20)
project(ShogiEngine)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCL REQUIRED)

file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(shogi ${SOURCES})
target_include_directories(shogi PRIVATE src)
target_link_libraries(shogi PRIVATE OpenCL::OpenCL)