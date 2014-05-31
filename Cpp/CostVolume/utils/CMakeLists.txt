cmake_minimum_required (VERSION 2.6)
set(CMAKE_BUILD_TYPE Debug) 
project(dtam)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/local_store/Qt/5.3/gcc_64/lib/cmake")# "/local_store/opencv-2.4.9/debug/unix-install/")
#message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -pg -ggdb")

find_package( OpenCV 2.4.9 REQUIRED )
find_package( Qt5Core REQUIRED)

#message(STATUS ${OpenCV_CONSIDERED_CONFIGS})


set(BASEPATH "${CMAKE_SOURCE_DIR}")
include_directories("${BASEPATH}")
include_directories(CostVolume)


add_executable (a.out testprog.cpp convertAhandaPovRayToStandard.cpp
CostVolume/Cost.cpp CostVolume/utils/reproject.cpp CostVolume/optimizer.part.cpp)

target_link_libraries( a.out ${OpenCV_LIBS} )
