cmake_minimum_required(VERSION 3.16)
project(untitled1 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(untitled1 main.cu common.h hello_cl.c cl_add.c)

set_target_properties(untitled1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(untitled1  cublas)