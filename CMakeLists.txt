cmake_minimum_required(VERSION 3.16)
project(DeepFeedforwardNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fnn_lib
    lib/src/FNN.cpp
    lib/src/Matrix.cpp
)

target_include_directories(fnn_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/include
)

add_executable(example
    examples/src/main.cpp
)

target_link_libraries(example PRIVATE fnn_lib)
