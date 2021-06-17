cmake_minimum_required(VERSION 3.0)
project(slam)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(BUILD_VISUALIZATIONS OFF CACHE BOOL "Build visualizations")
add_definitions(-DUSE_DBOW2)

set(SLAM_SOURCE_FILES
    slam_implementation.cpp
    mapdb.cpp
    mapper.cpp
    mapper_helpers.cpp
    bundle_adjuster.cpp
    keyframe.cpp
    map_point.cpp
    loop_ransac.cpp
    feature_search.cpp
    keyframe_matcher.cpp
    bow_index.cpp
    loop_closer.cpp
    relocation.cpp
    viewer_data_publisher.cpp
    optimize_transform.cpp
    id.cpp
    static_settings.cpp
    orb_extractor.cpp
    feature_detector.cpp
    image_pyramid.cpp
    # remaining unmodified OpenVSLAM files
    openvslam/essential_solver.cc
    openvslam/random_array.cc)

if (BUILD_VISUALIZATIONS)
    if (APPLE)
        find_package(GLEW 2.0 REQUIRED)
        list(APPEND EXTRA_INCLUDES ${GLEW_INCLUDE_DIR})
        list(APPEND EXTRA_LIBS ${GLEW_LIBRARY})
    endif()
    list(APPEND SLAM_SOURCE_FILES
      slam_viewer.cpp
      opencv_viewer_data_publisher.cpp)
endif()

add_library(slam ${SLAM_SOURCE_FILES})

# libmetis and suitesparseconfig have to be specified here due to linking
# issues caused by dropping GPL components from g2o / OpenVSLAM. Could be
# fixed properly later
target_include_directories(slam PUBLIC "." ${EXTRA_INCLUDES})
target_link_libraries(slam mobile-cv-suite::core mobile-cv-suite::slam parameters ${EXTRA_LIBS})
