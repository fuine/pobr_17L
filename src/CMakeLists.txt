set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -Wextra -pedantic -Wconversion -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -Wextra -pedantic -Wconversion -O2")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_library(ImageRecogn
            ImageRecogn.hpp
            Segmentation.cpp Segmentation.hpp
            ContourRecognition.cpp ContourRecognition.hpp
            Features.cpp Features.hpp
            Classifier.cpp Classifier.hpp
           )

target_link_libraries(ImageRecogn ${OpenCV_LIBS})
add_executable(asd main.cpp)
target_link_libraries(asd ImageRecogn)
