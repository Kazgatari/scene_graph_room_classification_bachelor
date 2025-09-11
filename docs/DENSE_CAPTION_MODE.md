# Dense Region Caption Processing Mode

The Spatial Object Tracker now supports a new processing mode for handling dense region captions from Florence2. Unlike traditional object detection, this mode processes DetectedObjects messages where the `label` field contains rich descriptive text instead of simple object names.

## Overview

This mode processes Florence2's DENSE_REGION_CAPTION output, which provides detailed natural language descriptions of detected regions. The system extracts the first two words for spatial duplicate detection and integrates with an LLM parser to extract structured object attributes.

## Florence2 DENSE_REGION_CAPTION Format

The system processes DetectedObjects messages where labels contain descriptive text like:
- `"black office chair in dimly lit room with desk and chair in background"`
- `"wooden table with books and papers scattered on surface"`
- `"large sofa covered with throw pillows in various colors"`

## Features

- **Unified Message Format**: Uses DetectedObjects for both OD and DENSE_CAPTION modes
- **Intelligent Label Processing**: Extracts first two words ("black office", "wooden table") for duplicate detection  
- **Spatial Duplicate Detection**: Prevents redundant LLM processing of similar objects
- **LLM Integration**: Sends full descriptions to LLM parser for structured attribute extraction
- **Resource Management**: Tracks pending LLM requests to avoid duplicate processing
- **Structured Output**: Extracts main_object, parts, environment, and spatial_context

## Configuration

### Launch Parameters

- `processing_mode`: Set to `"DENSE_CAPTION"` for dense caption mode or `"OD"` for object detection
- `position_threshold`: Spatial threshold for object matching (default: 1.5m)
- `confidence_threshold`: Confidence threshold for object tracking (default: 0.3)

### Example Launch Command

```bash
# Launch in dense caption mode
roslaunch scene_graph spatial_tracker_dense_caption.launch

# Or run manually with parameters
rosrun scene_graph spatial_object_tracker.py _processing_mode:=DENSE_CAPTION
```

## ROS Topics

### Input Topics

- `/detected_objects` (scene_graph/DetectedObjects): Detected objects from Florence2 (both OD and DENSE_CAPTION modes)
- `/scene_graph/parser_llm/result` (std_msgs/String): LLM parser results

### Output Topics

- `/scene_graph/parser_llm` (std_msgs/String): LLM parser requests
- `/scene_graph/seen_graph_objects` (scene_graph/GraphObjects): Processed objects for graph management

## Message Format

### Input DetectedObject (DENSE_CAPTION mode)
```
DetectedObject:
  label: "black office chair in dimly lit room with desk and chair in background"
  class_name: "chair"  # First word extracted
  confidence: 0.8
  position: {x: 1.2, y: 0.5, z: 0.0}
```

### LLM Parser Request
```json
{
    "object_index": 123,
    "caption": "wooden chair sitting in the corner...",
    "timestamp": 1234567890.123
}
```

### LLM Parser Response
```json
{
    "object_index": 123,
    "parsed_data": {
        "main_object": {
            "name": "chair",
            "material": "wooden",
            "color": "brown"
        },
        "parts": [
            {"name": "cushion", "color": "red"}
        ],
        "environment": [
            {"name": "corner", "relation": "in"}
        ],
        "spatial_context": {
            "location": "corner of room"
        }
    }
}
```

## Workflow

1. **DetectedObjects Reception**: Florence2 DENSE_REGION_CAPTION results received as DetectedObjects messages
2. **Mode Detection**: System checks `processing_mode` parameter to determine processing strategy
3. **Label Processing**: For DENSE_CAPTION mode, extracts first two words from full description
4. **Duplicate Detection**: Search for existing objects with similar label prefixes
5. **Object Management**:
   - If similar object exists and hasn't been processed → Send full description to LLM parser
   - If no similar objects → Create new object and send to LLM parser
6. **LLM Processing**: Full caption sent to LLM parser with object index
7. **Result Integration**: Parsed attributes stored in object instance
8. **Graph Publishing**: Updated objects published to graph management system

## Duplicate Detection Strategy

The system uses a two-word prefix matching strategy:
- Extract first two words from dense caption (e.g., "wooden chair" from "wooden chair sitting in corner...")
- Search existing objects for labels starting with this prefix
- Process only one object per similar label to avoid resource-heavy LLM calls

## Testing

Use the provided test script to validate the implementation:

```bash
# Run the test script
rosrun scene_graph test_dense_caption.py

# Choose from:
# 1. Automatic test with sample DetectedObjects containing dense captions
# 2. Interactive test (manual caption input as DetectedObjects)
# 3. Monitor mode (just observe results)
```

### Manual Testing with rostopic

```bash
# Create a DetectedObjects message with dense caption
rostopic pub /detected_objects scene_graph/DetectedObjects "
header:
  stamp: now
objects:
- label: 'black office chair in dimly lit room with desk in background'
  class_name: {data: 'chair'}
  confidence: 0.8
  position: {x: 1.0, y: 0.5, z: 0.0}
"
```

## Integration with LLM Parser

Ensure the LLM parser node is running with compatible message formats:

```bash
rosrun scene_graph parser_llm.py _model_path:=/models/Mistral-7B-Instruct-v0.3-AWQ
```

The LLM parser should:
- Subscribe to `/scene_graph/parser_llm` for requests
- Publish results to `/scene_graph/parser_llm/result`
- Include object_index in responses for proper matching

## Performance Considerations

- **LLM Request Tracking**: System tracks pending requests to prevent duplicate processing
- **Spatial Indexing**: Uses KDTree for efficient spatial searches
- **Label-based Filtering**: First-two-words strategy reduces search complexity
- **Background Processing**: LLM processing doesn't block main tracking loop

## Troubleshooting

### Common Issues

1. **No LLM Responses**: Check if parser_llm node is running and subscribed to correct topics
2. **Duplicate Processing**: Verify object_index matching in LLM responses
3. **Memory Usage**: Monitor LLM parser resource usage for large caption batches

### Debug Information

Enable debug logging to see detailed processing:
```bash
rosrun scene_graph spatial_object_tracker.py _processing_mode:=DENSE_CAPTION --log-level DEBUG
```

### Monitoring

- Check `/scene_graph/seen_graph_objects` for processed objects
- Monitor `/scene_graph/parser_llm` for LLM requests
- Watch logs for duplicate detection and object creation messages
