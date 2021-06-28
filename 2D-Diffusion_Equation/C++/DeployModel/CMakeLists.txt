cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(DeployModel)
set(CMAKE_PREFIX_PATH /home/sunny/Downloads/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(DeployModel main.cpp)
target_link_libraries(DeployModel "${TORCH_LIBRARIES}")
set_property(TARGET DeployModel PROPERTY CXX_STANDARD 14)



