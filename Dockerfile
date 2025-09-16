# Use ROS Noetic with full desktop (includes rviz, rqt, etc.)
FROM osrf/ros:noetic-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV CATKIN_WS=/catkin_ws

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Python development
    python3-pip \
    python3-dev \
    python3-setuptools \
    # ROS build tools
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    # Additional ROS packages
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2 \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    # Computer vision libraries
    #libopencv-dev \
    #python3-opencv \
    # Other utilities
    #vim \
    nano \
    #htop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    # Core ML/CV libraries
    numpy==1.21.6 \
    opencv-python==4.8.1.78 \
    # Computer vision and ML
    #torch==1.13.1 \
    #torchvision==0.14.1 \
    #ultralytics==8.0.196 \
    # Graph processing
    networkx==2.8.8 \
    # Geometry processing
    shapely==1.8.5 \
    # API and web requests
    requests==2.28.2 \
    # Google AI services
    google-generativeai==0.3.2 \
    # Image processing
    Pillow==9.5.0 \
    # Scientific computing
    scipy==1.9.3 \
    scikit-learn==1.1.3 \
    # Data handling
    #pandas==1.5.3

# Create catkin workspace
RUN mkdir -p $CATKIN_WS/src
WORKDIR $CATKIN_WS

# Initialize rosdep
RUN rosdep init || echo "rosdep already initialized" \
    && rosdep update

# Copy your source code
COPY . $CATKIN_WS/src/scene_graph_room_classification_bachelor/

# Install ROS dependencies for your package
RUN cd $CATKIN_WS \
    && rosdep install --from-paths src --ignore-src -r -y

# Build the catkin workspace
RUN cd $CATKIN_WS \
    && /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"

# Set up environment
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc \
    && echo "source $CATKIN_WS/devel/setup.bash" >> ~/.bashrc

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/ros/noetic/setup.bash\n\
source $CATKIN_WS/devel/setup.bash\n\
exec "$@"' > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]