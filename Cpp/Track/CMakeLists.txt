include_directories(utils)
find_package( OpenCV 2.4 REQUIRED )
find_package(Boost COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})
add_library (TrackFiles SHARED
                Track.cpp 
                align.part.cpp)
target_link_libraries( TrackFiles  ${OpenCV_LIBS} ${Boost_LIBRARIES})
