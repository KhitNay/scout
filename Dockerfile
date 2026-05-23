FROM dustynv/ros:humble-ros-base-l4t-r35.4.1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor > /usr/share/keyrings/ros-archive-keyring.gpg \
    && apt-get update && apt-get install -y \
      curl gnupg2 git cmake build-essential \
      libusb-1.0-0-dev libboost-python-dev libssl-dev \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
RUN cd /opt && git clone --depth 1 https://github.com/IntelRealSense/librealsense.git \
    && cd librealsense && mkdir build && cd build \
    && cmake .. \
      -DFORCE_RSUSB_BACKEND=true \
      -DBUILD_WITH_CUDA=true \
      -DBUILD_EXAMPLES=false \
      -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && rm -rf /opt/librealsense
RUN . /opt/ros/humble/install/setup.sh \
    && mkdir -p /ros2_ws/src \
    && cd /ros2_ws/src \
    && git clone --depth 1 https://github.com/realsenseai/realsense-ros.git -b ros2-master \
    && cd /ros2_ws \
    && colcon build --packages-select realsense2_camera_msgs realsense2_description \
    && rm -rf /ros2_ws/src/realsense-ros
ENV ROS2_WS=/ros2_ws
