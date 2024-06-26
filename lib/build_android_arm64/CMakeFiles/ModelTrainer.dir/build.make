# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/andrey/Desktop/mnist_tf/lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/andrey/Desktop/mnist_tf/lib/build_android_arm64

# Include any dependencies generated for this target.
include CMakeFiles/ModelTrainer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ModelTrainer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ModelTrainer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ModelTrainer.dir/flags.make

CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o: CMakeFiles/ModelTrainer.dir/flags.make
CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o: /Users/andrey/Desktop/mnist_tf/lib/ModelTrainer.cpp
CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o: CMakeFiles/ModelTrainer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/andrey/Desktop/mnist_tf/lib/build_android_arm64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o"
	/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ --target=aarch64-none-linux-android26 --gcc-toolchain=/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64 --sysroot=/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o -MF CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o.d -o CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o -c /Users/andrey/Desktop/mnist_tf/lib/ModelTrainer.cpp

CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.i"
	/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ --target=aarch64-none-linux-android26 --gcc-toolchain=/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64 --sysroot=/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/andrey/Desktop/mnist_tf/lib/ModelTrainer.cpp > CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.i

CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.s"
	/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ --target=aarch64-none-linux-android26 --gcc-toolchain=/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64 --sysroot=/Users/andrey/Library/Android/sdk/ndk/21/toolchains/llvm/prebuilt/darwin-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/andrey/Desktop/mnist_tf/lib/ModelTrainer.cpp -o CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.s

# Object files for target ModelTrainer
ModelTrainer_OBJECTS = \
"CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o"

# External object files for target ModelTrainer
ModelTrainer_EXTERNAL_OBJECTS =

libModelTrainer.so: CMakeFiles/ModelTrainer.dir/ModelTrainer.cpp.o
libModelTrainer.so: CMakeFiles/ModelTrainer.dir/build.make
libModelTrainer.so: /Users/andrey/Desktop/mnist_tf/lib/arm64-v8a/libtensorflowlite.so
libModelTrainer.so: /Users/andrey/Desktop/mnist_tf/lib/arm64-v8a/libtensorflowlite_flex.so
libModelTrainer.so: CMakeFiles/ModelTrainer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/andrey/Desktop/mnist_tf/lib/build_android_arm64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libModelTrainer.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ModelTrainer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ModelTrainer.dir/build: libModelTrainer.so
.PHONY : CMakeFiles/ModelTrainer.dir/build

CMakeFiles/ModelTrainer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ModelTrainer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ModelTrainer.dir/clean

CMakeFiles/ModelTrainer.dir/depend:
	cd /Users/andrey/Desktop/mnist_tf/lib/build_android_arm64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/andrey/Desktop/mnist_tf/lib /Users/andrey/Desktop/mnist_tf/lib /Users/andrey/Desktop/mnist_tf/lib/build_android_arm64 /Users/andrey/Desktop/mnist_tf/lib/build_android_arm64 /Users/andrey/Desktop/mnist_tf/lib/build_android_arm64/CMakeFiles/ModelTrainer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ModelTrainer.dir/depend

