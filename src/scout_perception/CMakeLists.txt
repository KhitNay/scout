cmake_minimum_required(VERSION 3.10)
project(scout_perception)

# Use C++17 — required for structured bindings and std::filesystem
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Treat warnings as errors during development
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# ROS2 dependencies (found via ament / CMake find_package)
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)

# ---------------------------------------------------------------------------
# OpenCV — standard system install
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
find_package(CUDA REQUIRED)

# ---------------------------------------------------------------------------
# TensorRT — JetPack installs headers to /usr/include/aarch64-linux-gnu
# and libraries to /usr/lib/aarch64-linux-gnu
# ---------------------------------------------------------------------------
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIB_DIR     /usr/lib/aarch64-linux-gnu)

if(NOT EXISTS "${TENSORRT_LIB_DIR}/libnvinfer.so")
  message(FATAL_ERROR "TensorRT not found. Expected: ${TENSORRT_LIB_DIR}/libnvinfer.so")
endif()

# ---------------------------------------------------------------------------
# detection_node executable
# ---------------------------------------------------------------------------
add_executable(detection_node src/detection_node.cpp)

# Include ONNX Runtime headers
target_include_directories(detection_node PRIVATE
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  include
)

# Link against all dependencies
ament_target_dependencies(detection_node
  rclcpp
  sensor_msgs
  cv_bridge
  realsense2_camera_msgs
)

target_link_libraries(detection_node
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIB_DIR}/libnvinfer.so
  ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
)

# ---------------------------------------------------------------------------
# Install rules — colcon needs these to find the built binary at runtime
# ---------------------------------------------------------------------------
install(TARGETS detection_node
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Linting (optional, skipped by default — remove AMENT_LINT flags to enable)
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
