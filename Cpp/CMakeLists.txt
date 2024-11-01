cmake_minimum_required (VERSION 2.6)
set(CMAKE_BUILD_TYPE Debug) 
project(dtam)
cmake_policy(SET CMP0012 NEW)
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/local_store/Qt/5.3/gcc_64/lib/cmake")# "/local_store/opencv-2.4.9/debug/unix-install/")
#message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb ")

find_package( OpenCV 2.4 REQUIRED )
find_package( Qt5Core REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system thread)
#add_definitions( -D__OPENCV_BUILD=1 )
#message(STATUS ${OpenCV_CONSIDERED_CONFIGS})

macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND DTAM_SOURCES "${_relPath}/${_src}")
        else()
            list (APPEND DTAM_SOURCES "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (DTAM_SOURCES ${DTAM_SOURCES} PARENT_SCOPE)
    endif()
endmacro(add_sources)



set(BASEPATH "${CMAKE_SOURCE_DIR}")
 

#add_library(GraphicsLib SHARED graphics.cpp)#must be in own library so we can link against it.


add_subdirectory(CostVolume)
add_subdirectory(Track)
add_subdirectory(Optimizer)
add_subdirectory(DepthmapDenoiseWeightedHuber)
add_subdirectory(utils)
add_subdirectory(Scheduler)


include_directories("${BASEPATH}" ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# Future proofing nvcc code by forcing PTX (device independent) code to be generated instead of gpu specific cubin code, from:
# https://devtalk.nvidia.com/default/topic/938165/when-will-cuda-8-be-released-1080-can-t-run-with-7-5/
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=compute_30; -gencode arch=compute_35,code=compute_35;--use_fast_math ;--restrict; -lineinfo; -O3)
cuda_add_library(OpenDTAM SHARED ${DTAM_SOURCES} OpenDTAM.cpp)
target_link_libraries(OpenDTAM opencv_gpu)
add_executable (a.out testmapper.cpp convertAhandaPovRayToStandard.cpp graphics.cpp)
target_link_libraries( a.out  OpenDTAM ${OpenCV_LIBS} ${Boost_LIBRARIES})

