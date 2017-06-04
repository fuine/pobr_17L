set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -Wall -Wextra -pedantic -Wconversion -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -Wall -Wextra -pedantic -Wconversion -g")

find_package (Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories (${Pobr_SOURCE_DIR}/src
                     ${Boost_INCLUDE_DIRS}
                     )
add_definitions (-DBOOST_TEST_DYN_LINK)

add_executable (TestSegmentation test_segmentation.cpp)

add_library(test_links INTERFACE)
target_link_libraries (test_links
        INTERFACE
        ImageRecogn
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
        )

target_link_libraries (TestSegmentation test_links)


add_test (NAME TestSegmentation WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND TestSegmentation)

file(COPY "test_files" DESTINATION "${CMAKE_BINARY_DIR}")
