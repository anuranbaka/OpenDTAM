

include_directories(utils ${BASEPATH} )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35;--use_fast_math ;--restrict; -lineinfo)


#cuda_compile(COST_VOLUME_O SHARED CostVolume.cu )
cuda_add_library (CostFiles SHARED
                Cost.cpp 
                utils/reproject.cpp 
                optimizer.part.cpp
                utils/reprojectCloud.cpp
                CostVolume.cpp CostVolume.cu
                ../gpu/ArrayTexture.cpp)
target_link_libraries( CostFiles  ${OpenCV_LIBS} ${Boost_LIBRARIES})

