cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)
project(opencl_wrapper)

add_executable(opencl_wrapper main.cpp opencl_wrapper.cpp)

if(APPLE)
    target_link_libraries(opencl_wrapper "-framework OpenCL")
elseif(UNIX)
    target_link_libraries(opencl_wrapper OpenCL)
endif()