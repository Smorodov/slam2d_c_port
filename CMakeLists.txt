cmake_minimum_required (VERSION 3.27)
project (main C CXX)

# Uncomment below to build and run in Unicode
add_definitions(-DUNICODE -D_UNICODE)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DEBUG_POSTFIX _d)

find_package( OpenCV REQUIRED )
if( OpenCV_FOUND )
#list( APPEND ThirdParty_LIBS ${OpenCV_LIBS} )
    include_directories( ${OpenCV_INCLUDE_DIRS} )
endif( OpenCV_FOUND )


find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# add the binary tree to the search path for include files
include_directories("${PROJECT_BINARY_DIR}")
add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -DWIN32 -D__STDC__)
set( headers 

)

set( sources 
main.cpp
)

add_executable(${PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${PROJECT_NAME}  Eigen3::Eigen ${OpenCV_LIBS})
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

