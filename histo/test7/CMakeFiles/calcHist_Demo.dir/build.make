# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pg210/Desktop/histo/test7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pg210/Desktop/histo/test7

# Include any dependencies generated for this target.
include CMakeFiles/calcHist_Demo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/calcHist_Demo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/calcHist_Demo.dir/flags.make

CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o: CMakeFiles/calcHist_Demo.dir/flags.make
CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o: calcHist_Demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/pg210/Desktop/histo/test7/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o -c /home/pg210/Desktop/histo/test7/calcHist_Demo.cpp

CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/pg210/Desktop/histo/test7/calcHist_Demo.cpp > CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.i

CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/pg210/Desktop/histo/test7/calcHist_Demo.cpp -o CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.s

CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.requires:
.PHONY : CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.requires

CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.provides: CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.requires
	$(MAKE) -f CMakeFiles/calcHist_Demo.dir/build.make CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.provides.build
.PHONY : CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.provides

CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.provides.build: CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o

# Object files for target calcHist_Demo
calcHist_Demo_OBJECTS = \
"CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o"

# External object files for target calcHist_Demo
calcHist_Demo_EXTERNAL_OBJECTS =

calcHist_Demo: CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o
calcHist_Demo: CMakeFiles/calcHist_Demo.dir/build.make
calcHist_Demo: /usr/local/lib/libopencv_videostab.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_video.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_ts.a
calcHist_Demo: /usr/local/lib/libopencv_superres.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_stitching.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_photo.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_ocl.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_objdetect.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_nonfree.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_ml.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_legacy.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_imgproc.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_highgui.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_gpu.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_flann.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_features2d.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_core.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_contrib.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_calib3d.so.2.4.10
calcHist_Demo: /usr/lib/x86_64-linux-gnu/libGLU.so
calcHist_Demo: /usr/lib/x86_64-linux-gnu/libGL.so
calcHist_Demo: /usr/lib/x86_64-linux-gnu/libSM.so
calcHist_Demo: /usr/lib/x86_64-linux-gnu/libICE.so
calcHist_Demo: /usr/lib/x86_64-linux-gnu/libX11.so
calcHist_Demo: /usr/lib/x86_64-linux-gnu/libXext.so
calcHist_Demo: /usr/local/lib/libopencv_nonfree.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_ocl.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_gpu.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_photo.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_objdetect.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_legacy.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_video.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_ml.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_calib3d.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_features2d.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_highgui.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_imgproc.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_flann.so.2.4.10
calcHist_Demo: /usr/local/lib/libopencv_core.so.2.4.10
calcHist_Demo: CMakeFiles/calcHist_Demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable calcHist_Demo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calcHist_Demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/calcHist_Demo.dir/build: calcHist_Demo
.PHONY : CMakeFiles/calcHist_Demo.dir/build

CMakeFiles/calcHist_Demo.dir/requires: CMakeFiles/calcHist_Demo.dir/calcHist_Demo.cpp.o.requires
.PHONY : CMakeFiles/calcHist_Demo.dir/requires

CMakeFiles/calcHist_Demo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/calcHist_Demo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/calcHist_Demo.dir/clean

CMakeFiles/calcHist_Demo.dir/depend:
	cd /home/pg210/Desktop/histo/test7 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pg210/Desktop/histo/test7 /home/pg210/Desktop/histo/test7 /home/pg210/Desktop/histo/test7 /home/pg210/Desktop/histo/test7 /home/pg210/Desktop/histo/test7/CMakeFiles/calcHist_Demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/calcHist_Demo.dir/depend

