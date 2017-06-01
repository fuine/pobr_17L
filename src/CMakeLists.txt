set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(ImageRecogn
            ImageRecogn.cpp ImageRecogn.hpp
           )

add_executable(asd main.cpp)
target_link_libraries(asd ImageRecogn ${OpenCV_LIBS})
