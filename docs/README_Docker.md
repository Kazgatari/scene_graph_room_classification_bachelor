# Scene Graph Room Classification - Docker Setup

## Quick Start with Docker

### 1. Build the Docker image
```bash
cd /path/to/your/project
docker build -t scene-graph-pipeline .
```

### 2. Run with Docker Compose (Recommended)
```bash
# Start the container
docker-compose up -d

# Access the container
docker-compose exec scene_graph_pipeline bash

# Inside container, run your nodes
roscore &
rosrun scene_graph vlm_service_node.py &
rosrun scene_graph graph_management_node.py &
```

### 3. Alternative: Run directly with Docker
```bash
# For GUI applications (rviz)
xhost +local:docker

# Run container with GUI support
docker run -it \
  --network host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd):/catkin_ws/src/scene_graph_room_classification_bachelor \
  scene-graph-pipeline
```

## Dependencies Included

### Base Image
- **osrf/ros:noetic-desktop-full**: Complete ROS Noetic with desktop tools

### System Packages
- Build tools: gcc, cmake, git
- ROS packages: cv_bridge, tf, message_filters, visualization_msgs
- OpenCV libraries and Python bindings

### Python Packages
- **Core ML/Vision**: numpy, opencv-python, torch, ultralytics
- **Graph Processing**: networkx (for scene graphs)
- **Geometry**: shapely (for spatial calculations) 
- **APIs**: requests, google-generativeai (for VLM services)
- **Science**: scipy, scikit-learn, pandas
- **Image**: Pillow

## Configuration

### Environment Variables
Set these in your Docker environment or docker-compose.yml:

```bash
# Gemini API (for external VLM backend)
GOOGLE_API_KEY=your_gemini_api_key

# VLM Backend selection
VLM_BACKEND=external  # or 'florence', 'vLLM'

# Model paths
YOLO_MODEL_PATH=/models/yolov9e-seg.pt
RF_CLASSIFIER_PATH=/models/rf_classifier.pkl
```

### Volume Mounts
- Source code: Mount for development
- Models: Mount model files directory
- X11: For GUI applications (rviz)

## Running Individual Nodes

```bash
# Inside the container
source /catkin_ws/devel/setup.bash

# Start roscore
roscore &

# VLM Service (with Gemini backend)
rosrun scene_graph vlm_service_node.py \
  _backend:=external \
  _api_key:=$GOOGLE_API_KEY

# Graph Management
rosrun scene_graph graph_management_node.py

# Room Classification  
rosrun scene_graph room_classification_node.py

# Visual Interface
rosrun scene_graph visual_interface_base.py
```

## Troubleshooting

### Common Issues

1. **GUI Applications (rviz) not working**
   ```bash
   xhost +local:docker  # Run on host
   ```

2. **ROS Communication issues**
   - Ensure `network_mode: "host"` in docker-compose.yml
   - Check ROS_MASTER_URI and ROS_HOSTNAME

3. **Permission issues with mounted volumes**
   ```bash
   # Fix ownership in container
   chown -R $(id -u):$(id -g) /catkin_ws
   ```

4. **Missing model files**
   - Mount your models directory to `/models` in container
   - Update paths in configuration files