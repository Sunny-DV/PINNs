cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(neural)
set(CMAKE_PREFIX_PATH /home/sunny/Downloads/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(neural main.cpp)
target_link_libraries(neural "${TORCH_LIBRARIES}")
set_property(TARGET neural PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
