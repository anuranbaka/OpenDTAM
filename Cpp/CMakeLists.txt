cmake_minimum_required (VERSION 2.6)
project(dtam)
find_package( OpenCV REQUIRED )
add_subdirectory(CostVolume)
include_directories(CostVolume)
add_executable(a.out testprog.cpp convertAhandaPovRayToStandard.cpp
CostVolume/reproject.part.cpp )
target_link_libraries( a.out ${OpenCV_LIBS} )
