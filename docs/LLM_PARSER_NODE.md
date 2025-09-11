# LLM Parser ROS Node

A high-performance ROS node for parsing dense scene descriptions using Large Language Models (LLMs) with batch processing and intelligent queuing.

## Overview

The `parser_llm.py` node integrates with the `spatial_object_tracker.py` to provide structured parsing of dense region captions from Florence2. It uses vLLM for efficient batch processing and automatic retry mechanisms.

## Key Features

### 🚀 **High-Performance Architecture**
- **Batch Processing**: Processes multiple requests together for 2-5x throughput improvement
- **vLLM Integration**: Optimized inference engine with GPU acceleration
- **Intelligent Queuing**: Dynamic batching with configurable timeouts
- **Automatic Retries**: Robust error handling with retry mechanisms

### 🔧 **Configurable Parameters**
- **Model Selection**: Support for various Qwen models (1.5B to 8B parameters)
- **Batch Size**: Adjustable batch size (1-16 requests per batch)
- **Timeout Control**: Configurable batch timeout for responsiveness
- **Memory Management**: GPU memory utilization control

### 📊 **Monitoring & Statistics**
- **Real-time Statistics**: Success rates, processing times, queue status
- **Performance Metrics**: Automatic reporting every 30 seconds
- **Error Tracking**: Detailed logging and failure analysis

## Installation

### Dependencies
```bash
pip install vllm torch transformers
```

### ROS Package
The node is part of the `scene_graph` package and should be built with catkin:
```bash
cd /root/catkin_ws && catkin_make
```

## Usage

### Basic Launch
```bash
# Launch LLM parser alone
roslaunch scene_graph parser_llm.launch

# Launch complete system (spatial tracker + LLM parser)
roslaunch scene_graph dense_caption_system.launch
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"Qwen/Qwen2.5-1.5B-Instruct"` | HuggingFace model identifier |
| `batch_size` | `4` | Number of requests to process together |
| `batch_timeout` | `2.0` | Max wait time (seconds) to form a batch |
| `gpu_memory_utilization` | `0.8` | GPU memory usage (0.1-0.9) |
| `max_retries` | `2` | Number of retry attempts for failed requests |

### Example Launch with Custom Parameters
```xml
<node pkg="scene_graph" type="parser_llm.py" name="parser_llm" output="screen">
    <param name="model_name" value="Qwen/Qwen3-4B-AWQ" />
    <param name="batch_size" value="8" />
    <param name="batch_timeout" value="1.5" />
    <param name="gpu_memory_utilization" value="0.9" />
</node>
```

## ROS Communication

### Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/scene_graph/parser_llm` | `std_msgs/String` | Input | Parsing requests from spatial tracker |
| `/scene_graph/parser_llm/result` | `std_msgs/String` | Output | Parsed results sent back to spatial tracker |

### Message Formats

#### Input Request Format
```json
{
    "object_index": 123,
    "caption": "black office chair in dimly lit room with desk in background",
    "timestamp": 1234567890.123
}
```

#### Output Result Format
```json
{
    "object_index": 123,
    "timestamp": 1234567890.456,
    "success": true,
    "parsed_data": {
        "main_object": {
            "name": "office_chair",
            "attributes": {
                "color": "black",
                "material": "",
                "style": "office"
            }
        },
        "parts": [],
        "environment": [
            {
                "name": "room",
                "attributes": {
                    "lighting": "dimly_lit"
                }
            },
            {
                "name": "desk",
                "attributes": {}
            }
        ],
        "spatial_context": {
            "position": "in room with desk in background",
            "nearby_objects": ["desk"]
        }
    }
}
```

## Performance Optimization

### Model Selection Guidelines

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ⚡⚡⚡ | ⭐⭐ | Fast prototyping, low-resource |
| `Qwen/Qwen3-4B-AWQ` | 4B | ⚡⚡ | ⭐⭐⭐ | Balanced performance |
| `Qwen/Qwen3-8B` | 8B | ⚡ | ⭐⭐⭐⭐ | High-quality parsing |

### Batch Processing Tuning

**High Throughput Setup:**
```xml
<param name="batch_size" value="8" />
<param name="batch_timeout" value="3.0" />
```

**Low Latency Setup:**
```xml
<param name="batch_size" value="2" />
<param name="batch_timeout" value="0.5" />
```

### Memory Management
- **GPU Memory**: Start with 0.8, increase to 0.9 if stable
- **System RAM**: Ensure 4GB+ available for model loading
- **Model Caching**: Models are cached after first load

## Monitoring

### Statistics Output
```
📊 LLM Parser Statistics:
   Total requests: 150
   Successful: 145
   Failed: 5
   Success rate: 96.7%
   Current queue size: 3
```

### Performance Metrics
- **Batch Processing Time**: Typically 0.5-2.0s per batch
- **Individual Request Time**: 0.1-0.5s average
- **Throughput**: 10-50 requests/second depending on configuration

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Reduce gpu_memory_utilization to 0.6-0.7
   ```

2. **Model Loading Failures**
   ```
   Solution: Check internet connection and HuggingFace access
   ```

3. **JSON Parsing Errors**
   ```
   Solution: Model may need fine-tuning or different sampling parameters
   ```

### Debug Mode
Enable debug logging:
```bash
rosrun scene_graph parser_llm.py --log-level DEBUG
```

### Performance Profiling
Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Integration with Spatial Object Tracker

The LLM parser seamlessly integrates with the spatial object tracker:

1. **Request Flow**: Spatial tracker sends caption parsing requests
2. **Batch Processing**: LLM parser queues and processes requests in batches
3. **Result Delivery**: Parsed results sent back with structured JSON
4. **Error Handling**: Failed requests automatically retried or reported

### Example Integration Workflow
```python
# Spatial tracker sends request
request = {
    "object_index": 42,
    "caption": "wooden chair with red cushion in corner",
    "timestamp": time.time()
}
self.parser_llm_request_pub.publish(json.dumps(request))

# LLM parser processes and responds
result = {
    "object_index": 42,
    "success": true,
    "parsed_data": {
        "main_object": {"name": "chair", "attributes": {"material": "wooden"}},
        "parts": [{"name": "cushion", "attributes": {"color": "red"}}],
        # ... more structured data
    }
}
```

## Future Enhancements

- **Multi-GPU Support**: Distribute processing across multiple GPUs
- **Model Hot-Swapping**: Switch models without restarting
- **Adaptive Batching**: Dynamic batch size based on queue length
- **Persistent Caching**: Save parsed results to avoid reprocessing
- **Custom Prompts**: Configurable system prompts for different domains
