cmake_minimum_required(VERSION 3.10.0)

project(structure_from_motion)

set(exe_name sfm)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

if (NOT OpenCV_LIBS)
  set(OpenCV_LIBS "opencv::opencv")
endif()

add_exectuable(${exe_name}
  sift.cc
  filter.cc
  )

target_link_libraries(${exe_name}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  )

# Define two groups of supported compilers.
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

# Set compiler specific flags and definitions.
target_compile_options(${exe_name} PRIVATE
  "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wformat=2>>"
  "$<${msvc_cxx}:$<BUILD_INTERFACE:-W4>>"
  )
target_compile_definitions(${exe_name} PUBLIC
  "$<${msvc_cxx}:-D_USE_MATH_DEFINES>"
  )
