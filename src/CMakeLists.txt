cmake_minimum_required(VERSION 3.10.0)

project(structure_from_motion)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)
find_package(tek5030 CONFIG REQUIRED OPTIONAL_COMPONENTS rs2)

if (NOT OpenCV_LIBS)
  set(OpenCV_LIBS "opencv::opencv")
endif()

add_executable(sift main_sift.cpp
        sift.cc
        filters.cc
        #epipolar_geo.h
        #epipolar_geo.cpp
)

target_link_libraries(sift PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        tek5030::tek5030
)


# Define two groups of supported compilers.
set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

# Set compiler specific flags and definitions for sift.
target_compile_options(sift PRIVATE
        "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wformat=2>>"
        "$<${msvc_cxx}:$<BUILD_INTERFACE:-W4>>"
)
target_compile_definitions(sift PUBLIC
        "$<${msvc_cxx}:-D_USE_MATH_DEFINES>"
)

# Set compiler specific flags and definitions for dual_example.

#SET_TARGET_PROPERTIES(camera PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET_TARGET_PROPERTIES(sift PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#set(CMAKE_CXX_FLAGS "-O3")