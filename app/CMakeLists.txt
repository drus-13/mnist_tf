cmake_minimum_required(VERSION 3.13)
project(UseModelTrainerApp)

# Add the executable
add_executable(UseModelTrainerApp main.cpp)

# find_package(boinc CONFIG REQUIRED)

# target_link_libraries(UseModelTrainerApp
#     unofficial::boinc::boinc
#     unofficial::boinc::boincapi
# )

# Create an imported target for libtensorflowlite.so
add_library(libtensorflowlite SHARED IMPORTED)
set_target_properties(libtensorflowlite PROPERTIES
    IMPORTED_LOCATION "/Users/andrey/Desktop/mnist_tf/lib/arm64-v8a/libtensorflowlite.so"
    INTERFACE_INCLUDE_DIRECTORIES "/Users/andrey/Desktop/mnist_tf/lib/include"
)

# Create an imported target for libtensorflowlite_flex.so
add_library(libtensorflowlite_flex SHARED IMPORTED)
set_target_properties(libtensorflowlite_flex PROPERTIES
    IMPORTED_LOCATION "/Users/andrey/Desktop/mnist_tf/lib/arm64-v8a/libtensorflowlite_flex.so"
)

# Do the same for libModelTrainer.so, and specify its dependency on libtensorflowlite
add_library(libModelTrainer SHARED IMPORTED)
set_target_properties(libModelTrainer PROPERTIES
    IMPORTED_LOCATION "/Users/andrey/Desktop/mnist_tf/lib/build_android_arm64/libModelTrainer.so"
    INTERFACE_INCLUDE_DIRECTORIES "/Users/andrey/Desktop/mnist_tf/lib"
)
target_link_libraries(libModelTrainer 
    INTERFACE
    libtensorflowlite 
    -Wl,--no-as-needed # Need --no-as-needed to link tensorflowlite_flex 
    libtensorflowlite_flex
)

# Link UseModelTrainerApp with libModelTrainer only, libtensorflowlite will be included automatically
target_link_libraries(UseModelTrainerApp
    -Wl,--no-as-needed
    libModelTrainer
)

# If the library uses TensorFlow Lite or other Android NDK libraries, link them as well
find_library(log-lib log)
target_link_libraries(UseModelTrainerApp ${log-lib})