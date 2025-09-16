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
    # Image processing (needed by: vlm_service_node.py, visual_interface_base.py)
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    # Sensor data synchronization (needed by: visual_interface_base.py)
    ros-noetic-message-filters \
    # Coordinate transformations (needed by: graph_management_node.py, visual_interface_base.py)
    ros-noetic-tf \
    ros-noetic-tf-conversions \
    ros-noetic-tf2 \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    # Visualization markers (needed by: graph_management_node.py for rviz)
    ros-noetic-visualization-msgs \
    # Basic message types (needed by: all nodes)
    ros-noetic-geometry-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    # Navigation messages (needed by: visual_interface_base.py for odometry)
    ros-noetic-nav-msgs

# Install OpenCV (needed by: vlm_service_node.py, visual_interface_base.py)
echo "Installing OpenCV..."
sudo apt-get install -y \
    libopencv-dev

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user \
    # Core scientific computing (needed by: all nodes)
    numpy==1.21.6 \
    # Computer vision (needed by: vlm_service_node.py, visual_interface_base.py)
    opencv-python==4.8.1.78 \
    # HTTP requests (needed by: vlm_service_node.py for API calls)
    requests==2.28.2 \
    # Google Gemini API (needed by: vlm_service_node.py for external VLM backend)
    google-generativeai==0.3.2 \
    # Image processing (needed by: vlm_service_node.py for Gemini API)
    Pillow==9.5.0 \
    # Scientific computing (needed by: graph_management_node.py for calculations)
    scipy==1.9.3 \
    # Machine learning (needed by: room_classification_node.py for RF classifier)
    scikit-learn==1.1.3 \
    # Data manipulation (needed by: room_classification_node.py, classification_utils.py)
    pandas==1.5.3 \
    # Graph processing (needed by: graph_management_node.py for scene graph structure)
    networkx==2.8.8 \
    # Geometry processing (needed by: graph_management_node.py for polygon operations)
    shapely==1.8.5
    
    # Commented out - only needed if using YOLO or advanced features:
    #torch==1.13.1 \              # PyTorch (needed by: YOLO models if used)
    #torchvision==0.14.1 \        # PyTorch vision (needed by: YOLO models if used)
    #ultralytics==8.0.196 \       # YOLO (needed by: ros_yolo_node.py if used)
    
    

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
