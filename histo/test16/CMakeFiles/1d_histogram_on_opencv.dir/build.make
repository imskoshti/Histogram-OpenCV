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
CMAKE_SOURCE_DIR = /home/pg210/Desktop/histo/test16

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pg210/Desktop/histo/test16

# Include any dependencies generated for this target.
include CMakeFiles/1d_histogram_on_opencv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/1d_histogram_on_opencv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/1d_histogram_on_opencv.dir/flags.make

CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o: CMakeFiles/1d_histogram_on_opencv.dir/flags.make
CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o: 1d_histogram_on_opencv.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/pg210/Desktop/histo/test16/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o -c /home/pg210/Desktop/histo/test16/1d_histogram_on_opencv.cpp

CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/pg210/Desktop/histo/test16/1d_histogram_on_opencv.cpp > CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.i

CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/pg210/Desktop/histo/test16/1d_histogram_on_opencv.cpp -o CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.s

CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.requires:
.PHONY : CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.requires

CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.provides: CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.requires
	$(MAKE) -f CMakeFiles/1d_histogram_on_opencv.dir/build.make CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.provides.build
.PHONY : CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.provides

CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.provides.build: CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o

# Object files for target 1d_histogram_on_opencv
1d_histogram_on_opencv_OBJECTS = \
"CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o"

# External object files for target 1d_histogram_on_opencv
1d_histogram_on_opencv_EXTERNAL_OBJECTS =

1d_histogram_on_opencv: CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o
1d_histogram_on_opencv: CMakeFiles/1d_histogram_on_opencv.dir/build.make
1d_histogram_on_opencv: /usr/local/lib/libopencv_videostab.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_video.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_ts.a
1d_histogram_on_opencv: /usr/local/lib/libopencv_superres.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_stitching.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_photo.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_ocl.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_objdetect.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_nonfree.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_ml.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_legacy.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_imgproc.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_highgui.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_gpu.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_flann.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_features2d.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_core.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_contrib.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_calib3d.so.2.4.10
1d_histogram_on_opencv: /usr/lib/x86_64-linux-gnu/libGLU.so
1d_histogram_on_opencv: /usr/lib/x86_64-linux-gnu/libGL.so
1d_histogram_on_opencv: /usr/lib/x86_64-linux-gnu/libSM.so
1d_histogram_on_opencv: /usr/lib/x86_64-linux-gnu/libICE.so
1d_histogram_on_opencv: /usr/lib/x86_64-linux-gnu/libX11.so
1d_histogram_on_opencv: /usr/lib/x86_64-linux-gnu/libXext.so
1d_histogram_on_opencv: /usr/local/lib/libopencv_nonfree.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_ocl.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_gpu.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_photo.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_objdetect.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_legacy.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_video.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_ml.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_calib3d.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_features2d.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_highgui.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_imgproc.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_flann.so.2.4.10
1d_histogram_on_opencv: /usr/local/lib/libopencv_core.so.2.4.10
1d_histogram_on_opencv: CMakeFiles/1d_histogram_on_opencv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable 1d_histogram_on_opencv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/1d_histogram_on_opencv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/1d_histogram_on_opencv.dir/build: 1d_histogram_on_opencv
.PHONY : CMakeFiles/1d_histogram_on_opencv.dir/build

CMakeFiles/1d_histogram_on_opencv.dir/requires: CMakeFiles/1d_histogram_on_opencv.dir/1d_histogram_on_opencv.cpp.o.requires
.PHONY : CMakeFiles/1d_histogram_on_opencv.dir/requires

CMakeFiles/1d_histogram_on_opencv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/1d_histogram_on_opencv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/1d_histogram_on_opencv.dir/clean

CMakeFiles/1d_histogram_on_opencv.dir/depend:
	cd /home/pg210/Desktop/histo/test16 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pg210/Desktop/histo/test16 /home/pg210/Desktop/histo/test16 /home/pg210/Desktop/histo/test16 /home/pg210/Desktop/histo/test16 /home/pg210/Desktop/histo/test16/CMakeFiles/1d_histogram_on_opencv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/1d_histogram_on_opencv.dir/depend

