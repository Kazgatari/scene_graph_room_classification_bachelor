# VLM Object Detection Pipeline

This module provides a complete pipeline for detecting objects in images using Vision-Language Models (VLMs) and converting them to 3D scene graphs with spatial information.

## Architecture

```
Image + Depth + Odometry → VLM Service → Object Detection → 3D Positioning → GraphObjects
```

## Components

### 1. VisualInterfaceBase
- **Purpose**: Base class for synchronized sensor data processing
- **Input**: Camera image, depth point cloud, odometry data
- **Output**: GraphObjects messages with 3D positioned objects
- **Features**:
  - Camera intrinsics handling
  - 2D to 3D coordinate transformation
  - Depth sampling within bounding boxes
  - Object size estimation from depth and bounding box
  - VLM service integration

### 2. VLM Service Node
- **Purpose**: Provides object detection using different VLM backends
- **Backends**: Florence, vLLM, External APIs (Gemini)
- **Input**: ROS Image message
- **Output**: ObjectSceneGraph array with detected objects

### 3. VLM Object Detection Node
- **Purpose**: Complete object detection pipeline
- **Functionality**: Combines sensor data processing with VLM inference

## Usage

### 1. Start the VLM Service

```bash
# Florence backend (fastest)
rosrun scene_graph_room_classification_bachelor vlm_service_node.py _backend:=florence

# vLLM backend (good balance)
rosrun scene_graph_room_classification_bachelor vlm_service_node.py _backend:=vLLM

# External API (most accurate, requires API key)
rosrun scene_graph_room_classification_bachelor vlm_service_node.py _backend:=external _api_key:=your_key
```

### 2. Start the Object Detection Pipeline

```bash
# Using the complete launch file
roslaunch scene_graph_room_classification_bachelor vlm_object_detection_pipeline.launch backend:=florence

# Or individual node
rosrun scene_graph_room_classification_bachelor vlm_object_detection_node.py
```

### 3. Required Topics

**Input Topics:**
- `/camera/color/image_raw` (sensor_msgs/Image)
- `/camera/depth/points` (sensor_msgs/PointCloud2)
- `/odom` (nav_msgs/Odometry)
- `/camera/color/camera_info` (sensor_msgs/CameraInfo)

**Output Topics:**
- `/scene_graph/seen_graph_objects` (GraphObjects)

## 3D Positioning

### Coordinate Transformation Pipeline

1. **2D Detection**: VLM provides 2D bounding boxes in percentile coordinates (0-999)
2. **Depth Sampling**: Sample depth values within bounding box from point cloud
3. **Camera Coordinates**: Convert 2D pixel + depth to 3D camera coordinates
4. **World Coordinates**: Transform using robot odometry to world frame
5. **Size Estimation**: Calculate 3D object dimensions from 2D bbox and depth

### Size Estimation Heuristics

```python
# Object-specific size estimation
if aspect_ratio > 2.0:      # Wide objects (tables, shelves)
    depth = min(width, height) * 0.5
elif aspect_ratio < 0.5:    # Tall objects (bottles, lamps)  
    depth = width * 0.8
else:                       # Square objects
    depth = min(width, height) * 0.7
```

### Validation

- Bounding boxes must be positive and within image bounds
- World positions must be finite (no NaN/inf)
- Object sizes must be reasonable (5cm - 5m range)

## Message Structure

### GraphObject
```
std_msgs/String name                    # Object label
std_msgs/Int32 object_id               # Unique ID
geometry_msgs/Point32[] bounding_box   # Min/max corners [2 points]
```

### GraphObjects
```
std_msgs/Header header          # Timestamp and frame
GraphObject[] objects          # Array of detected objects
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | string | "florence" | VLM backend type |
| `florence_url` | string | "http://florence:8001" | Florence container URL |
| `vlm_url` | string | "http://localhost:8000/v1" | vLLM server URL |
| `api_key` | string | "" | External API key |

## Example Usage in Code

```python
#!/usr/bin/env python3
import rospy
from visual_interface_base import VisualInterfaceBase

class MyDetectionNode(VisualInterfaceBase):
    def __init__(self):
        super().__init__()
        # Your custom initialization
        
    def synchronized_callback(self, image_msg, depth_msg, odom_msg):
        # This will automatically:
        # 1. Call VLM service with image
        # 2. Calculate 3D positions
        # 3. Estimate object sizes  
        # 4. Publish GraphObjects
        super().synchronized_callback(image_msg, depth_msg, odom_msg)
        
        # Add your custom processing here

if __name__ == '__main__':
    rospy.init_node('my_detection_node')
    node = MyDetectionNode()
    rospy.spin()
```

## Docker Integration

### Example docker-compose.yml

```yaml
services:
  # VLM backend
  florence_vlm:
    build: ./florence_container
    container_name: florence
    ports: ["8001:8001"]
    networks: [vlm_network]

  # ROS pipeline  
  ros_detection:
    build: ./ros_container
    networks: [vlm_network]
    environment:
      - ROS_MASTER_URI=http://roscore:11311
    depends_on: [roscore, florence_vlm]
    command: >
      bash -c "
      source /opt/ros/noetic/setup.bash &&
      roslaunch scene_graph_room_classification_bachelor vlm_object_detection_pipeline.launch backend:=florence
      "
```

## Performance Notes

- **Processing Rate**: Typically 1-5 Hz depending on VLM backend
- **Memory Usage**: ~1-2GB for VLM service + models
- **Accuracy**: Depends on VLM model quality and depth data accuracy
- **Latency**: Florence (~200ms), vLLM (~1-3s), External (variable)

## Troubleshooting

1. **No objects detected**: Check VLM service is running and responsive
2. **Invalid coordinates**: Verify camera calibration and depth data quality
3. **Service connection failed**: Ensure VLM service node is started first
4. **Poor size estimates**: Check depth data quality and camera intrinsics

## Future Improvements

- Object tracking across frames
- Improved size estimation using object recognition
- Multi-view fusion for better accuracy
- Integration with semantic mapping
- Performance optimization for real-time use