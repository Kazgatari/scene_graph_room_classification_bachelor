#!/bin/bash

# ROS Scene Graph Room Classification - Dependency Installation Script
# This script installs all required dependencies for the pipeline

set -e

echo "Installing system dependencies..."

# Update package lists
sudo apt-get update

# Install build tools and development packages
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-setuptools

# Install ROS packages
echo "Installing ROS packages..."
sudo apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2 \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-visualization-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-nav-msgs

# Install OpenCV
echo "Installing OpenCV..."
sudo apt-get install -y \
    libopencv-dev

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user \
    numpy \
    opencv-python \
    requests \
    google-generativeai \
    Pillow \
    scipy \
    scikit-learn \
    pandas \
    networkx \
    shapely

# Commented out - only needed if using YOLO or advanced features:
# torch==1.13.1 - PyTorch (needed by: YOLO models if used)
# torchvision==0.14.1 - PyTorch vision (needed by: YOLO models if used)  
# ultralytics==8.0.196 - YOLO (needed by: ros_yolo_node.py if used)
    
    

# Install catkin tools if not already installed
if ! command -v catkin &> /dev/null; then
    echo "Installing catkin tools..."
    sudo apt-get install -y python3-catkin-tools
fi

echo "All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Build your catkin workspace:"
echo "   cd ~/catkin_ws && catkin_make"
echo ""
echo "2. Source your workspace:"
echo "   source ~/catkin_ws/devel/setup.bash"
echo ""
echo "3. Run your nodes:"
echo "   rosrun scene_graph vlm_service_node.py"
echo "   rosrun scene_graph graph_management_node.py"
