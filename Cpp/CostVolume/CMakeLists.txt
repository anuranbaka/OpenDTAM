include_directories(utils)
add_library (CostFiles SHARED
                Cost.cpp 
                utils/reproject.cpp 
                optimizer.part.cpp
                utils/reprojectCloud.cpp)

