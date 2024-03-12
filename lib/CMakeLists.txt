cmake_minimum_required(VERSION 3.10)
project(ModelTrainerLib VERSION 1.0)

# Specify where to find the header files for TF Lite C++
set(INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/include
        ${CMAKE_CURRENT_LIST_DIR}/include/flatbuffers/include)
include_directories(${INCLUDE_DIRS})

add_library(tflite SHARED IMPORTED)
add_library(tflite-flex SHARED IMPORTED)
set_target_properties(
    tflite PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/arm64-v8a/libtensorflowlite.so
)
set_target_properties(
    tflite-flex PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/arm64-v8a/libtensorflowlite_flex.so
)

# Source files
set(SOURCES
    ModelTrainer.cpp
)

# Build the main target `ModelTrainer` that will use TF Lite
# Create shared library from sources
add_library(ModelTrainer SHARED ${SOURCES})

# Link the main target with required libs `libtensorflowlite.so`
target_link_libraries(ModelTrainer 
    tflite
    -Wl,--no-as-needed # Need --no-as-needed to link tensorflowlite_flex 
    tflite-flex
)


