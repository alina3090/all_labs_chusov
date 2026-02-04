cmake_minimum_required(VERSION 3.22)
project(Parallelism)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)

add_executable(Parallelism
        entrypoint.cpp
        mod_ops.cpp
        num_threads.cpp
        performance.cpp
        randomize.cpp
        test.cpp
        vector_mod.cpp
)

if(OpenMP_FOUND)
    target_compile_options(Parallelism PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(Parallelism PUBLIC ${OpenMP_CXX_FLAGS})
    target_link_libraries(Parallelism PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found - using fallback")
endif()