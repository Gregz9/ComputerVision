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
        epipolar_geo.h
        epipolar_geo.cpp
)

add_executable(camera camera.cpp
        calibrated_camera.h
        calibrated_opencv_camera.h
        calibrated_opencv_camera.cpp
        frame.h
        frame.cpp
        sift.cc
        filters.cc
        intrinsic_calibration.h
        intrinsic_calibration.cpp
        epipolar_geo.h
        epipolar_geo.cpp
)

target_link_libraries(sift PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        tek5030::tek5030
)

if(TARGET tek5030::rs2)
  message(STATUS "* Support for Intel® RealSense™ ENABLED")
  target_link_libraries(camera PRIVATE tek5030::rs2 ${OpenCV_LIBS} tek5030::tek5030 Eigen3::Eigen)
  target_compile_definitions(camera PUBLIC TEK5030_REALSENSE)
  target_sources(camera PRIVATE
          calibrated_realsense_camera.h
          calibrated_realsense_camera.cpp
  )
else()
  message(STATUS "* Support for Intel® RealSense™ DISABLED because component tek5030::rs2 was not found")
endif()

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
target_compile_options(camera PRIVATE
        "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wformat=2>>"
        "$<${msvc_cxx}:$<BUILD_INTERFACE:-W4>>"
)
target_compile_definitions(camera PUBLIC
        "$<${msvc_cxx}:-D_USE_MATH_DEFINES>"
)
SET_TARGET_PROPERTIES(camera PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
SET_TARGET_PROPERTIES(sift PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#set(CMAKE_CXX_FLAGS "-O3")