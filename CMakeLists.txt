cmake_minimum_required(VERSION 3.1)
project(Pobr)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

add_subdirectory (src)

enable_testing ()
add_subdirectory (test)
