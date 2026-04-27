cmake_minimum_required(VERSION 3.8)
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
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(realsense2_camera_msgs REQUIRED)

# ---------------------------------------------------------------------------
# OpenCV — standard system install
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ---------------------------------------------------------------------------
# ONNX Runtime — installed manually to /usr/local
# We point CMake at the headers and library directly because there is no
# official FindOnnxRuntime.cmake provided by Microsoft for system installs.
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_INCLUDE_DIR /usr/local/include)
set(ONNXRUNTIME_LIB         /usr/local/lib/libonnxruntime.so)

# Fail early with a helpful message if ONNX Runtime is not installed.
if(NOT EXISTS "${ONNXRUNTIME_LIB}")
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "  Expected library : ${ONNXRUNTIME_LIB}\n"
    "  See CLAUDE.md for install instructions.")
endif()

# Wrap it in an IMPORTED target so we can link cleanly
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

# ---------------------------------------------------------------------------
# detection_node executable
# ---------------------------------------------------------------------------
add_executable(detection_node src/detection_node.cpp)

# Include ONNX Runtime headers
target_include_directories(detection_node PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
  include
)

# Link against all dependencies
ament_target_dependencies(detection_node
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
  realsense2_camera_msgs
)

target_link_libraries(detection_node
  ${OpenCV_LIBS}
  onnxruntime
)

# ---------------------------------------------------------------------------
# Install rules — colcon needs these to find the built binary at runtime
# ---------------------------------------------------------------------------
install(TARGETS detection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Linting (optional, skipped by default — remove AMENT_LINT flags to enable)
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
