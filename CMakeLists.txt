cmake_minimum_required(VERSION 3.28)
project(PaddleOCR)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "D:/project/cpp/opencv/build")
set(ONNXRUNTIME_DIR "D:/project/cpp/onnxruntime-win-x64-gpu-1.20.1")
add_definitions(-std=c++20)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# setup CUDA
find_package(CUDAToolkit  REQUIRED)
message(STATUS "    include path: ${CUDAToolkit_INCLUDE_DIRS}")
include_directories(${CUDAToolkit_INCLUDE_DIRS})
#include_directories(${Dirent_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
        NO_MODULE
        NO_DEFAULT_PATH
        NO_CMAKE_PATH
        NO_CMAKE_ENVIRONMENT_PATH
        NO_SYSTEM_ENVIRONMENT_PATH
        NO_CMAKE_PACKAGE_REGISTRY
        NO_CMAKE_BUILDS_PATH
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_DIR}/include)

# 打印 ONNXRUNTIME_DIR 以确保其正确设置
message("ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

file(GLOB ONNXRUNTIME_LIBS
        #        ${ONNXRUNTIME_DIR}/lib/*.dll
        ${ONNXRUNTIME_DIR}/lib/*.lib
)

# 打印找到的文件数量
list(LENGTH ONNXRUNTIME_LIBS ONNXRUNTIME_LIBS_COUNT)
message("Number of DLLs found: ${ONNXRUNTIME_LIBS_COUNT}")

if(ONNXRUNTIME_LIBS_COUNT GREATER 0)
    list(JOIN ONNXRUNTIME_LIBS "\n" ONNXRUNTIME_LIBS_STR)
    message("ONNXRUNTIME_LIBS:\n${ONNXRUNTIME_LIBS_STR}")
else()
    message("No DLLs found in the specified directory.")
endif()

add_executable(PaddleOCR main.cpp
        utils.cpp
        utils.h
        operators.cpp
        operators.h
        predict_base.cpp
        predict_base.h
        predict_det.cpp
        predict_det.h
        db_postprocess.cpp
        db_postprocess.h
        clipper.hpp
        clipper.cpp
        predict_rec.cpp
        predict_rec.h
        rec_postprocess.cpp
        rec_postprocess.h
        OCR.cpp
        OCR.h
)
#add_library(PaddleOCR SHARED
#        main.cpp
#        utils.cpp
#        utils.h
#        operators.cpp
#        operators.h
#        predict_base.cpp
#        predict_base.h
#        predict_det.cpp
#        predict_det.h
#        db_postprocess.cpp
#        db_postprocess.h
#        clipper.hpp
#        clipper.cpp
#        predict_rec.cpp
#        predict_rec.h
#        rec_postprocess.cpp
#        rec_postprocess.h
#        OCR.cpp
#        OCR.h
#)
target_link_libraries(PaddleOCR ${OpenCV_LIBS})
target_link_libraries(PaddleOCR ${CUDA_LIBRARIES})
target_link_libraries(PaddleOCR ${ONNXRUNTIME_LIBS})