cmake_minimum_required(VERSION 3.21)
project(SONN)

set(CMAKE_CXX_STANDARD 14)

add_executable(SONN dpi.c)

target_link_libraries(SONN pthread)