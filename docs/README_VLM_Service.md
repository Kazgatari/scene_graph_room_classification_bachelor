# VLM Service Node

A ROS service node that provides image analysis capabilities using different Vision-Language Model (VLM) backends. The service receives an image and returns an array of `ObjectSceneGraph` messages containing detected objects with their attributes and relationships.

## Features

- **Multiple Backend Support**: Florence, vLLM, and external APIs (e.g., Gemini)
- **Standardized Output**: Returns consistent `ObjectSceneGraph` messages regardless of backend
- **Docker-Ready**: Designed to work with containerized VLM services
- **Configurable**: Easy parameter-based configuration for different backends

## Backend Options

### 1. Florence (`backend: "florence"`)
- Uses a local Florence container
- Requires only the image as input
- Lightweight and fast
- Default URL: `http://florence:8001`

### 2. vLLM (`backend: "vLLM"`)
- Uses vLLM server with OpenAI-compatible API
- Supports advanced VLMs like Qwen2-VL
- Default URL: `http://localhost:8000/v1`
- Uses structured prompts for object detection

### 3. External (`backend: "external"`)
- Uses external APIs (e.g., Google Gemini)
- Requires API key authentication
- Useful for testing against cloud services

## Service Definition

```ros
# Request
sensor_msgs/Image image

---

# Response
ObjectSceneGraph[] scene_graphs
bool success
string error_message
```

## Usage

### 1. Basic Launch

```bash
# Florence backend (default)
roslaunch scene_graph_room_classification_bachelor vlm_service.launch

# Or with specific parameters
rosrun scene_graph_room_classification_bachelor vlm_service_node.py _backend:=florence

# vLLM backend
rosrun scene_graph_room_classification_bachelor vlm_service_node.py _backend:=vLLM _vlm_url:=http://localhost:8000/v1

# External API (requires API key)
rosrun scene_graph_room_classification_bachelor vlm_service_node.py _backend:=external _api_key:=your_api_key_here
```

### 2. Service Call Example

```python
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from scene_graph_room_classification_bachelor.srv import VLMInference

# Initialize ROS node
rospy.init_node('vlm_client')

# Wait for service
rospy.wait_for_service('vlm_inference')

# Create service proxy
vlm_service = rospy.ServiceProxy('vlm_inference', VLMInference)

# Call service with image
response = vlm_service(image_msg)

if response.success:
    for scene_graph in response.scene_graphs:
        print(f"Object: {scene_graph.main_object.name}")
        print(f"Color: {scene_graph.main_object.attributes.color}")
        print(f"Style: {scene_graph.main_object.attributes.style}")
else:
    print(f"Error: {response.error_message}")
```

### 3. Test Client

```bash
# Run the test client
rosrun scene_graph_room_classification_bachelor test_vlm_service.py
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | string | "florence" | Backend type: "florence", "vLLM", or "external" |
| `api_key` | string | "" | API key for external services (required for external backend) |
| `vlm_url` | string | "http://localhost:8000/v1" | URL for vLLM server |
| `florence_url` | string | "http://florence:8001" | URL for Florence container |
| `external_model` | string | "gemini-pro-vision" | Model name for external API |

## Docker Integration

### Docker Compose Example

```yaml
version: '3.8'

services:
  # ROS container with VLM service
  ros_pipeline:
    build: ./ros_container
    networks:
      - vlm_network
    environment:
      - ROS_MASTER_URI=http://roscore:11311
    depends_on:
      - roscore
      - florence_vlm

  # Florence VLM container
  florence_vlm:
    build: ./florence_container
    container_name: florence
    ports:
      - "8001:8001"
    networks:
      - vlm_network

  # vLLM container (alternative)
  vllm_server:
    build: ./vllm_container
    ports:
      - "8000:8000"
    networks:
      - vlm_network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

networks:
  vlm_network:
    driver: bridge
```

## Message Structure

The service returns `ObjectSceneGraph` messages with the following structure:

```
ObjectSceneGraph:
  ObjectInfo main_object:
    int32 id
    string name
    ObjectAttribute attributes:
      string color
      string style
  ObjectPart[] parts
  ObjectInfo[] environment
  ObjectSpatialContext spatial_context:
    string position
    string[] nearby_objects
```

## Detection Prompt

The service uses a structured prompt for vLLM and external backends:

```
Analyze this image and detect all objects. You must respond with ONLY valid JSON...

Required JSON format:
{
  "objects": [
    {
      "bounding_box": [x_min, y_min, x_max, y_max],
      "label": "object_name",
      "attributes": {
        "color": "color_value",
        "style": "style_value"
      },
      "relations": [
        {
          "type": "relation_type",
          "target": "related_object_name"
        }
      ]
    }
  ]
}
```

## Building

Make sure to add the service to your `CMakeLists.txt`:

```cmake
add_service_files(
  FILES
  VLMInference.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  scene_graph_room_classification_bachelor
)
```

## Troubleshooting

1. **Service not found**: Make sure the node is running and ROS core is active
2. **Backend connection failed**: Check that the specified backend container/service is running and accessible
3. **JSON parsing errors**: Check the VLM output format and adjust the prompt if needed
4. **Image conversion errors**: Ensure the input image message is valid and properly formatted

## Performance Notes

- **Florence**: Fastest, suitable for real-time applications
- **vLLM**: Good balance of speed and accuracy, GPU-accelerated
- **External**: Highest accuracy but dependent on network latency and API limits