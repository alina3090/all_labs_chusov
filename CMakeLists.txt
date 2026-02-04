cmake_minimum_required(VERSION 3.22)
project(Parallelism)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp")

add_executable(Parallelism
        main.cpp
)


