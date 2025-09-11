#!/usr/bin/env python3

# CRITICAL: Set multiprocessing method and environment BEFORE any imports that might initialize CUDA
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # Suppress multiprocessing warning
os.environ['VLLM_LOGGING_LEVEL'] = 'WARNING'  # Reduce log verbosity
os.environ['VLLM_USE_V1'] = '0'  # Force V0 engine for better compatibility
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'  # Use xformers instead of flash-attn
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure single GPU usage
os.environ['VLLM_NO_USAGE_STATS'] = '1'  # Disable usage stats
os.environ['VLLM_DISABLE_FLASH_ATTN'] = '1'  # Completely disable flash attention

import multiprocessing
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import rospy
import cv2
import numpy as np
import json
import threading
import time
import signal
import sys
import atexit
import gc
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image

# ROS imports
from sensor_msgs.msg import Image as ROSImage, CompressedImage
from scene_graph.msg import ObjectSceneGraph, ObjectInfo, ObjectPart, ObjectSpatialContext, ObjectAttribute
from cv_bridge import CvBridge
import message_filters

# Defer vLLM imports to avoid segfaults during module loading
vllm_available = False
LLM = None
SamplingParams = None
TextPrompt = None
ImagePrompt = None

def safe_import_vllm():
    """Safely import vLLM with error handling."""
    global vllm_available, LLM, SamplingParams, TextPrompt, ImagePrompt
    
    try:
        rospy.loginfo("Attempting to import vLLM...")
        
        # Set environment variables for safer CUDA initialization
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")  # Force xformers backend
        
        # Import with timeout protection
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available - vLLM requires GPU")
        
        rospy.loginfo(f"CUDA available: {torch.cuda.is_available()}")
        rospy.loginfo(f"CUDA devices: {torch.cuda.device_count()}")
        rospy.loginfo(f"Current device: {torch.cuda.current_device()}")
        
        # Clear any existing CUDA state
        torch.cuda.empty_cache()
        
        # Now import vLLM - try different import patterns for different versions
        from vllm import LLM, SamplingParams
        
        # Try to import input classes (may not exist in all versions)
        try:
            from vllm.inputs import TextPrompt, ImagePrompt
        except ImportError:
            # Fallback for older versions
            try:
                from vllm import TextPrompt, ImagePrompt
            except ImportError:
                # Set to None if not available
                TextPrompt = None
                ImagePrompt = None
                rospy.logwarn("TextPrompt/ImagePrompt not available - using basic interface")
        
        vllm_available = True
        rospy.loginfo("vLLM imported successfully")
        return True
        
    except ImportError as e:
        rospy.logerr(f"vLLM import failed: {e}")
        rospy.logerr("Please install vLLM: pip install vllm")
        return False
    except Exception as e:
        rospy.logerr(f"Error during vLLM import: {e}")
        rospy.logerr("vLLM initialization failed - falling back to dummy mode")
        return False

class QwenVLvLLMNode:
    """
    High-performance QwenVL node using vLLM for object detection and scene graph generation.
    Features batch processing, streaming inference, and optimized memory management.
    """
    
    def __init__(self):
        rospy.init_node('qwenvl_vllm_node', anonymous=True)
        
        # Initialize basic threading variables first (needed for cleanup)
        self.running = True
        self.shutdown_event = threading.Event()
        self.processing_queue = deque()
        self.queue_lock = threading.Lock()
        
        # Initialize parameters
        self.model_name = rospy.get_param('~model_name', 'Qwen/Qwen2-VL-7B-Instruct-AWQ')  # Try AWQ version first
        
        # Fallback models for better compatibility (AWQ first, then standard)
        self.fallback_models = [
            'Qwen/Qwen2-VL-7B-Instruct-AWQ',  # Primary - AWQ quantized version (often more stable)
            'Qwen/Qwen2-VL-2B-Instruct-AWQ',  # Smaller AWQ fallback
            'Qwen/Qwen2-VL-7B-Instruct',      # Standard version fallback
            'Qwen/Qwen2-VL-2B-Instruct',      # Smallest fallback
        ]
        self.detection_interval = rospy.get_param('~detection_interval', 1.0)
        self.max_objects = rospy.get_param('~max_objects', 50)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.6)
        self.batch_size = rospy.get_param('~batch_size', 1)  # Must be 1 for multimodal models with vLLM
        self.max_queue_size = rospy.get_param('~max_queue_size', 16)
        self.tensor_parallel_size = rospy.get_param('~tensor_parallel_size', 1)
        self.gpu_memory_utilization = rospy.get_param('~gpu_memory_utilization', 0.8)
        self.enable_streaming = rospy.get_param('~enable_streaming', False)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Try to import vLLM safely
        rospy.loginfo("Checking vLLM availability...")
        if not safe_import_vllm():
            rospy.logfatal("vLLM import failed - cannot continue")
            rospy.signal_shutdown("vLLM import failed")
            return
        
        # Initialize vLLM model with retry mechanism
        rospy.loginfo("Initializing vLLM model...")
        self.llm = None
        if not self.initialize_vllm_model_safe():
            rospy.logfatal("vLLM model initialization failed - cannot continue")
            rospy.signal_shutdown("vLLM model initialization failed")
            return
        
        # Create optimized sampling parameters for JSON generation
        self.sampling_params = SamplingParams(
            temperature=0.01,  # Very low for deterministic JSON output
            top_p=0.95,       # Slightly higher for better token diversity
            max_tokens=256,   # Reduced for compact JSON (you said always compact)
            top_k=50,         # Add top_k for more focused sampling
            stop=["</s>", "<|endoftext|>", "\n\nHuman:", "\n\nUser:", "```", "}\n\n"],  # Stop after JSON end
            repetition_penalty=1.05,  # Light penalty to avoid repetition
            length_penalty=0.95,      # Slight bias toward shorter responses
            skip_special_tokens=True,  # Clean output without special tokens
        )
        
        # Threading and batch management
        self.processing_queue = deque(maxlen=self.max_queue_size)
        self.result_callbacks = {}
        self.queue_lock = threading.Lock()
        self.request_id_counter = 0
        self.running = True  # Flag to control thread execution
        self.shutdown_event = threading.Event()  # Event for coordinated shutdown
        
        # Performance monitoring (disabled for now)
        # self.inference_times = deque(maxlen=100)
        # self.batch_stats = {
        #     'total_batches': 0,
        #     'total_images': 0,
        #     'avg_batch_time': 0.0,
        #     'avg_throughput': 0.0
        # }
        
        # Publishers
        self.result_pub = rospy.Publisher(
            '/scene_graph/parser_llm/result', 
            ObjectSceneGraph, 
            queue_size=10
        )
        
        # Subscribers with message filtering for synchronization
        self.image_sub = message_filters.Subscriber('/camera/image_raw', ROSImage)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', ROSImage)
        
        # Synchronize image and depth messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Start batch processing thread
        self.processing_thread = threading.Thread(target=self.batch_processing_loop, daemon=False)
        self.processing_thread.start()
        
        # Start performance monitoring thread
        #self.monitor_thread = threading.Thread(target=self.performance_monitor, daemon=False)
        #self.monitor_thread.start()
        
        # Register signal handlers and cleanup functions
        self.setup_signal_handlers()
        
        rospy.loginfo("QwenVL vLLM node initialized successfully!")
        rospy.loginfo(f"Model: {self.model_name}")
        rospy.loginfo(f"Batch size: {self.batch_size}")
        rospy.loginfo(f"Tensor parallel size: {self.tensor_parallel_size}")
        rospy.loginfo(f"Streaming enabled: {self.enable_streaming}")
    
    def initialize_vllm_model_safe(self):
        """Initialize the vLLM model with comprehensive safety checks and error handling."""
        try:
            rospy.loginfo("Starting safe vLLM initialization...")
            
            # Clear GPU memory before initialization
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    rospy.loginfo(f"GPU memory cleared")
                else:
                    rospy.logerr("CUDA not available")
                    return False
            except Exception as e:
                rospy.logwarn(f"Could not clear GPU memory: {e}")
                return False
            
            # Force garbage collection
            gc.collect()
            
            # Try different models in order of preference
            for model_name in self.fallback_models:
                rospy.loginfo(f"Trying model: {model_name}")
                
                # Initialize with conservative parameters optimized for your system
                init_params = {
                    "model": model_name,
                    "tensor_parallel_size": 1,  # Force single GPU
                    "gpu_memory_utilization": 0.7,  # Slightly higher for AWQ (they use less memory)
                    "max_model_len": 32768,  # Increased to handle multimodal tokens (32768 required)
                    "trust_remote_code": True,
                    "enforce_eager": True,  # Disable CUDA graphs for stability
                    "disable_log_stats": True,
                    "max_num_seqs": 1,  # Must be 1 to handle large multimodal sequences
                    "max_num_batched_tokens": 32768,  # Match max_model_len
                    "disable_custom_all_reduce": True,  # Better stability
                }
                
                # Handle quantization and dtype based on model name
                if "AWQ" in model_name:
                    # AWQ models - use marlin backend for better performance
                    init_params["quantization"] = "awq"  # Let vLLM auto-select the best AWQ backend
                    init_params["dtype"] = "half"  # AWQ works best with half precision
                    rospy.loginfo("Using AWQ quantization with half precision")
                elif "GPTQ" in model_name:
                    init_params["quantization"] = "gptq"
                    init_params["dtype"] = "half"
                    rospy.loginfo("Using GPTQ quantization")
                else:
                    # Non-quantized models - use auto for better compatibility
                    init_params["dtype"] = "auto"
                    rospy.loginfo("Using auto dtype for non-quantized model")
                
                try:
                    rospy.loginfo(f"Initializing vLLM with model {model_name}...")
                    rospy.loginfo(f"Init params: {init_params}")
                    
                    # Create LLM instance
                    self.llm = LLM(**init_params)
                    
                    # Update the model name to the successful one
                    self.model_name = model_name
                    
                    rospy.loginfo(f"✓ vLLM model {model_name} initialized successfully")
                    return True
                    
                except Exception as e:
                    rospy.logwarn(f"✗ Model {model_name} failed: {str(e)[:100]}...")
                    
                    # Clean up failed attempt
                    if hasattr(self, 'llm') and self.llm is not None:
                        try:
                            del self.llm
                            self.llm = None
                            gc.collect()
                            
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    
                    # Continue to next model
                    continue
            
            # If we get here, all models failed
            rospy.logerr("All fallback models failed to initialize")
            return False
                    
        except Exception as e:
            rospy.logerr(f"vLLM initialization failed: {e}")
            rospy.logerr(f"Error type: {type(e).__name__}")
            
            # Clean up failed attempt
            if hasattr(self, 'llm') and self.llm is not None:
                try:
                    del self.llm
                    self.llm = None
                    gc.collect()
                    
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            
            return False
    
    def check_gpu_environment(self):
        """Simple GPU environment check."""
        try:
            import torch
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except:
            return False
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            rospy.loginfo(f"Received signal {signal_name}, initiating graceful shutdown...")
            self.shutdown()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Register atexit handler for unexpected exits
        atexit.register(self.cleanup_on_exit)
    
    def synchronized_callback(self, image_msg: ROSImage, depth_msg: ROSImage):
        """Handle synchronized image and depth messages."""
        try:
            # Convert ROS images to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            # Add to processing queue with metadata
            request_data = {
                'request_id': self.request_id_counter,
                'timestamp': time.time(),
                'image': cv_image,
                'depth': cv_depth,
                'header': image_msg.header,
                'callback': self.publish_results
            }
            
            with self.queue_lock:
                if len(self.processing_queue) < self.max_queue_size:
                    self.processing_queue.append(request_data)
                    self.request_id_counter += 1
                else:
                    rospy.logwarn("Processing queue full, dropping frame")
                    
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")
    
    def batch_processing_loop(self):
        """Main batch processing loop for optimal throughput."""
        try:
            while self.running and not rospy.is_shutdown() and not self.shutdown_event.is_set():
                try:
                    batch_requests = []
                    
                    # Collect batch of requests
                    with self.queue_lock:
                        while len(batch_requests) < self.batch_size and self.processing_queue:
                            batch_requests.append(self.processing_queue.popleft())
                    
                    if not batch_requests:
                        # Short sleep to prevent busy waiting, but check shutdown frequently
                        if self.shutdown_event.wait(timeout=0.01):
                            break
                        continue
                    
                    # Process batch
                    start_time = time.time()
                    self.process_batch(batch_requests)
                    batch_time = time.time() - start_time
                    
                    # Simple logging without stats
                    rospy.logdebug(f"Processed batch of {len(batch_requests)} images in {batch_time:.3f}s")
                    
                except Exception as e:
                    if self.running:  # Only log if we're supposed to be running
                        rospy.logerr(f"Error in batch processing iteration: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            if self.running:  # Only log if we're supposed to be running
                rospy.logerr(f"Fatal error in batch processing loop: {e}")
        finally:
            rospy.loginfo("Batch processing thread terminated")
    
    def process_batch(self, batch_requests: List[Dict]):
        """Process a batch of image requests using vLLM."""
        try:
            # Prepare batch inputs
            prompts = []
            images = []
            
            for request in batch_requests:
                # Convert image to PIL format
                cv_image = request['image']
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Create detection prompt
                prompt = self.create_detection_prompt()
                
                # Create multimodal input
                prompts.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": pil_image}
                })
            
            # Run batch inference
            start_inference = time.time()
            
            if self.enable_streaming:
                # Use streaming for better responsiveness
                results = []
                for prompt_data in prompts:
                    outputs = self.llm.generate(
                        [prompt_data],
                        self.sampling_params,
                        use_tqdm=False
                    )
                    results.extend(outputs)
            else:
                # Standard batch processing
                results = self.llm.generate(
                    prompts,
                    self.sampling_params,
                    use_tqdm=False
                )
            
            inference_time = time.time() - start_inference
            rospy.logdebug(f"Inference completed in {inference_time:.3f}s for {len(prompts)} images")
            
            # Process results
            for i, (request, output) in enumerate(zip(batch_requests, results)):
                try:
                    generated_text = output.outputs[0].text
                    detected_objects = self.parse_detection_output(
                        generated_text, 
                        request['image'], 
                        request['depth']
                    )
                    
                    # Call the callback with results
                    request['callback'](detected_objects, request['header'])
                    
                except Exception as e:
                    rospy.logerr(f"Error processing result {i}: {e}")
            
        except Exception as e:
            rospy.logerr(f"Error in batch processing: {e}")
    
    def create_detection_prompt(self) -> str:
        """Create the detection prompt for the vision-language model."""
        return """
You are an advanced Multimodal AI tasked with analyzing an image to detect objects and their relationships. For each detected object, you must provide: 
Bounding Box: The coordinates of the object in the format [x_min, y_min, x_max, y_max]. 
Object Details: A compact JSON representation of the object, including: 
label: The name of the object. 
attributes: Key properties like color, material, and style. 
relations: Relationships to other objects (e.g., "attached_to", "on", "in", "under", "part_of"). 
Output Format: 
{
  "objects": [ 
    { "bounding_box": [x_min, y_min, x_max, y_max],
      "label": "object_name", 
      "attributes": { 
        "color": "color_value", 
        "material": "material_value", 
        "style": "style_value" }, 
      "relations": [ { 
        "type": "relation_type", 
        "target": "related_object_name" } ] 
    } ] 
}
Rules:
Include only attributes explicitly visible in the image. 
Use empty strings for missing attributes. 
Use procentile values for the bounding box, where the top left corner is [0,0] and the bottom right is [999,999]. 
Relations should describe spatial or structural relationships (e.g., "on", "under", "attached_to"). 
Ensure the JSON is compact and valid.

Task: Analyze the provided image and return the bounding boxes and JSON representation for all detected objects, following the format and rules above.
"""
    
    def parse_detection_output(
        self, 
        output_text: str, 
        image: np.ndarray, 
        depth: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Parse the model output and extract object detections."""
        try:
            # Find JSON in the output - look for both object format and array format
            json_data = None
            
            # Try to find JSON object format: {"objects": [...]}
            start_obj = output_text.find('{"objects"')
            if start_obj != -1:
                # Find the matching closing brace
                brace_count = 0
                end_obj = start_obj
                for i, char in enumerate(output_text[start_obj:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_obj = start_obj + i + 1
                            break
                
                json_str = output_text[start_obj:end_obj]
                try:
                    json_data = json.loads(json_str)
                    detections = json_data.get('objects', [])
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to find direct array format
            if json_data is None:
                start_idx = output_text.find('[')
                end_idx = output_text.rfind(']') + 1
                
                if start_idx == -1 or end_idx == 0:
                    rospy.logwarn("No valid JSON found in model output")
                    return []
                
                json_str = output_text[start_idx:end_idx]
                detections = json.loads(json_str)
            
            # Process each detection
            processed_objects = []
            height, width = image.shape[:2]
            
            for detection in detections:
                try:
                    # Validate and extract data
                    label = detection.get('label', 'unknown')
                    bbox = detection.get('bounding_box', [])
                    
                    # For new format, we don't have confidence in the output
                    # We'll assign a default confidence based on detection
                    confidence = 0.8  # Default confidence for detected objects
                    
                    # Handle both formats of bounding box
                    if len(bbox) >= 4:
                        # New format: [x_min, y_min, x_max, y_max] in percentiles (0-999)
                        x_min_pct, y_min_pct, x_max_pct, y_max_pct = bbox[:4]
                        
                        # Convert percentile coordinates (0-999) to pixel coordinates
                        x = max(0, min(width-1, int(x_min_pct * width / 999)))
                        y = max(0, min(height-1, int(y_min_pct * height / 999)))
                        x_max = max(x+1, min(width, int(x_max_pct * width / 999)))
                        y_max = max(y+1, min(height, int(y_max_pct * height / 999)))
                        
                        w = x_max - x
                        h = y_max - y
                    else:
                        rospy.logwarn(f"Invalid bounding box format: {bbox}")
                        continue
                    
                    # Calculate 3D position using depth
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Get depth value (handle different depth formats)
                    if depth.dtype == np.uint16:
                        depth_value = depth[center_y, center_x] / 1000.0  # Convert mm to meters
                    else:
                        depth_value = depth[center_y, center_x]
                    
                    # Extract attributes from new format
                    attributes_dict = detection.get('attributes', {})
                    attributes_list = []
                    
                    # Convert attributes dict to list format for compatibility
                    for key, value in attributes_dict.items():
                        if value and value.strip():  # Only include non-empty attributes
                            attributes_list.append(f"{key}: {value}")
                    
                    # Extract relations
                    relations = detection.get('relations', [])
                    relation_descriptions = []
                    for relation in relations:
                        rel_type = relation.get('type', '')
                        target = relation.get('target', '')
                        if rel_type and target:
                            relation_descriptions.append(f"{rel_type} {target}")
                    
                    # Create description from attributes and relations
                    description_parts = []
                    if attributes_list:
                        description_parts.append("Attributes: " + ", ".join(attributes_list))
                    if relation_descriptions:
                        description_parts.append("Relations: " + ", ".join(relation_descriptions))
                    description = "; ".join(description_parts)
                    
                    # Create processed object
                    obj_data = {
                        'label': label,
                        'confidence': confidence,
                        'bounding_box': [x, y, w, h],  # Keep as [x, y, width, height] for ROS compatibility
                        'center_3d': [center_x, center_y, depth_value],
                        'description': description,
                        'size': 'medium',  # Default size since not in new format
                        'attributes': attributes_list,
                        'relations': relations
                    }
                    
                    processed_objects.append(obj_data)
                    
                except Exception as e:
                    rospy.logwarn(f"Error processing detection: {e}")
                    continue
            
            rospy.loginfo(f"Detected {len(processed_objects)} objects")
            return processed_objects
            
        except json.JSONDecodeError as e:
            rospy.logerr(f"JSON parsing error: {e}")
            rospy.logdebug(f"Failed to parse JSON: {output_text[:500]}...")  # Log first 500 chars for debugging
            return []
        except Exception as e:
            rospy.logerr(f"Error parsing detection output: {e}")
            return []
    
    def publish_results(self, detected_objects: List[Dict], header):
        """Publish detected objects as ObjectSceneGraph message."""
        try:
            if not detected_objects:
                rospy.logwarn("No objects detected to publish")
                return
            
            # Create ObjectSceneGraph message for the most prominent object
            # (For simplicity, we'll use the first detected object as the main object)
            main_object_data = detected_objects[0]
            
            msg = ObjectSceneGraph()
            msg.header = header
            
            # Main object
            msg.main_object.id = 0
            msg.main_object.name = main_object_data['label']
            
            # Attributes for main object
            attributes_list = main_object_data.get('attributes', [])
            color = ""
            material = ""
            style = ""
            
            # Parse attributes from list format
            for attr in attributes_list:
                attr_lower = attr.lower()
                if 'color:' in attr_lower:
                    color = attr.split(':', 1)[1].strip()
                elif 'material:' in attr_lower:
                    material = attr.split(':', 1)[1].strip()
                elif 'style:' in attr_lower:
                    style = attr.split(':', 1)[1].strip()
            
            msg.main_object.attributes.color = color
            # material removed from ObjectAttribute
            msg.main_object.attributes.style = style
            
            # Parts - treat other detected objects as parts if they have relations to the main object
            relations = main_object_data.get('relations', [])
            main_obj_name = main_object_data['label']
            
            for obj_data in detected_objects[1:]:  # Skip the main object
                obj_relations = obj_data.get('relations', [])
                
                # Check if this object is related to the main object
                is_part = False
                relationship_type = ""
                
                for relation in obj_relations:
                    target = relation.get('target', '')
                    rel_type = relation.get('type', '')
                    if target.lower() == main_obj_name.lower() and rel_type in ['part_of', 'attached_to', 'component_of']:
                        is_part = True
                        relationship_type = rel_type
                        break
                
                if is_part:
                    part = ObjectPart()
                    part.name = obj_data['label']
                    part.relationship_to_main = relationship_type
                    
                    # Parse part attributes
                    part_attributes = obj_data.get('attributes', [])
                    part_color = ""
                    part_material = ""
                    part_style = ""
                    
                    for attr in part_attributes:
                        attr_lower = attr.lower()
                        if 'color:' in attr_lower:
                            part_color = attr.split(':', 1)[1].strip()
                        elif 'material:' in attr_lower:
                            part_material = attr.split(':', 1)[1].strip()
                        elif 'style:' in attr_lower:
                            part_style = attr.split(':', 1)[1].strip()
                    
                    part.attributes.color = part_color
                    # material removed
                    part.attributes.style = part_style
                    
                    msg.parts.append(part)
                else:
                    # Add as environment object
                    env_obj = ObjectInfo()
                    env_obj.id = len(msg.environment)
                    env_obj.name = obj_data['label']
                    
                    # Parse environment object attributes
                    env_attributes = obj_data.get('attributes', [])
                    env_color = ""
                    env_material = ""
                    env_style = ""
                    
                    for attr in env_attributes:
                        attr_lower = attr.lower()
                        if 'color:' in attr_lower:
                            env_color = attr.split(':', 1)[1].strip()
                        elif 'material:' in attr_lower:
                            env_material = attr.split(':', 1)[1].strip()
                        elif 'style:' in attr_lower:
                            env_style = attr.split(':', 1)[1].strip()
                    
                    env_obj.attributes.color = env_color
                    # material removed
                    env_obj.attributes.style = env_style
                    
                    msg.environment.append(env_obj)
            
            # Spatial context - extract spatial relationships from all objects
            nearby_objects = []
            spatial_relations = []
            
            for obj_data in detected_objects:
                for relation in obj_data.get('relations', []):
                    rel_type = relation.get('type', '')
                    target = relation.get('target', '')
                    if rel_type in ['on', 'under', 'near', 'in', 'above', 'below', 'beside'] and target:
                        spatial_relations.append(f"{obj_data['label']} {rel_type} {target}")
                        if target not in nearby_objects:
                            nearby_objects.append(target)
            
            msg.spatial_context.position = "; ".join(spatial_relations) if spatial_relations else ""
            msg.spatial_context.nearby_objects = nearby_objects
            
            # Publish the message
            self.result_pub.publish(msg)
            
            rospy.loginfo(f"Published ObjectSceneGraph with main object: {msg.main_object.name}, "
                         f"{len(msg.parts)} parts, {len(msg.environment)} environment objects")
                
        except Exception as e:
            rospy.logerr(f"Error publishing results: {e}")
            import traceback
            traceback.print_exc()
    
    def performance_monitor(self):
        """Monitor and log performance metrics."""
        try:
            while self.running and not rospy.is_shutdown() and not self.shutdown_event.is_set():
                try:
                    # Log every 30 seconds or until shutdown
                    if self.shutdown_event.wait(timeout=30.0):
                        break
                    
                    # Performance monitoring is currently disabled
                    # Uncomment the performance tracking variables above to enable
                    rospy.loginfo("Performance monitoring disabled - enable performance tracking variables to see stats")
                    
                    # if hasattr(self, 'inference_times') and self.inference_times:
                    #     avg_inference_time = np.mean(list(self.inference_times))
                    #     fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
                    #     
                    #     rospy.loginfo(
                    #         f"Performance Stats - "
                    #         f"Avg Inference: {avg_inference_time:.3f}s, "
                    #         f"FPS: {fps:.1f}, "
                    #         f"Batch Throughput: {self.batch_stats['avg_throughput']:.1f} img/s, "
                    #         f"Total Batches: {self.batch_stats['total_batches']}, "
                    #         f"Total Images: {self.batch_stats['total_images']}"
                    #     )
                    
                except Exception as e:
                    if self.running:  # Only log if we're supposed to be running
                        rospy.logerr(f"Error in performance monitor iteration: {e}")
        
        except Exception as e:
            if self.running:  # Only log if we're supposed to be running
                rospy.logerr(f"Fatal error in performance monitor: {e}")
        finally:
            rospy.loginfo("Performance monitor thread terminated")
    
    def run(self):
        """Main run loop."""
        rospy.loginfo("QwenVL vLLM node is running...")
        
        try:
            # Use rospy.spin() but catch interruption
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Received KeyboardInterrupt, shutting down...")
        except Exception as e:
            rospy.logerr(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of the node."""
        rospy.loginfo("Initiating graceful shutdown...")
        
        try:
            # Set shutdown flags
            self.running = False
            self.shutdown_event.set()
            
            # Stop ROS subscribers to prevent new data
            if hasattr(self, 'image_sub'):
                self.image_sub.unregister()
            if hasattr(self, 'depth_sub'):
                self.depth_sub.unregister()
            if hasattr(self, 'ts'):
                # Time synchronizer doesn't have unregister, but subscribers do
                pass
            
            rospy.loginfo("Stopped ROS subscribers")
            
            # Wait for threads to finish
            threads_to_join = []
            
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                threads_to_join.append(('processing', self.processing_thread))
            
            # Uncomment when performance monitoring is enabled
            # if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            #     threads_to_join.append(('monitor', self.monitor_thread))
            
            for thread_name, thread in threads_to_join:
                rospy.loginfo(f"Waiting for {thread_name} thread to finish...")
                thread.join(timeout=30.0)  # Increased timeout to 30 seconds
                if thread.is_alive():
                    rospy.logwarn(f"{thread_name} thread did not terminate within timeout, forcing termination...")
                    # Try to force terminate the thread (Python doesn't have direct thread killing)
                    try:
                        import ctypes
                        thread_id = thread.ident
                        if thread_id:
                            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(thread_id), ctypes.py_object(SystemExit)
                            )
                            if res > 1:
                                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                                rospy.logwarn(f"Failed to force terminate {thread_name} thread")
                            else:
                                rospy.loginfo(f"Force terminated {thread_name} thread")
                    except Exception as e:
                        rospy.logwarn(f"Could not force terminate {thread_name} thread: {e}")
                else:
                    rospy.loginfo(f"{thread_name} thread terminated successfully")
            
            # Clean up vLLM resources
            self.cleanup_vllm()
            
            rospy.loginfo("Graceful shutdown completed")
            
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup_on_exit(self):
        """Cleanup function called on unexpected exit."""
        try:
            if hasattr(self, 'running') and self.running:
                rospy.logwarn("Unexpected exit detected, performing emergency cleanup...")
                self.cleanup()
        except Exception as e:
            print(f"Error during emergency cleanup: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        rospy.loginfo("Cleaning up resources...")
        try:
            # Set shutdown flags
            if hasattr(self, 'running'):
                self.running = False
            if hasattr(self, 'shutdown_event'):
                self.shutdown_event.set()
            
            # Clean up vLLM
            self.cleanup_vllm()
            
            # Clear queues
            if hasattr(self, 'processing_queue'):
                with self.queue_lock if hasattr(self, 'queue_lock') else threading.Lock():
                    self.processing_queue.clear()
            
            rospy.loginfo("Cleanup completed")
            
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {e}")
    
    def cleanup_vllm(self):
        """Clean up vLLM model resources with aggressive cleanup."""
        try:
            if hasattr(self, 'llm') and self.llm is not None:
                rospy.loginfo("Cleaning up vLLM model with extended timeout...")
                
                # Try to properly destroy the vLLM engine with longer timeout
                try:
                    # First, try to gracefully stop any ongoing inference
                    rospy.loginfo("Attempting graceful vLLM shutdown...")
                    
                    # Clear any pending requests
                    if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'scheduler'):
                        try:
                            # Try to clear the scheduler if possible
                            scheduler = self.llm.llm_engine.scheduler
                            if hasattr(scheduler, 'running'):
                                scheduler.running = False
                        except Exception as e:
                            rospy.logwarn(f"Could not access scheduler: {e}")
                    
                    # Wait a bit for any ongoing operations to complete
                    time.sleep(2.0)
                    
                    # Now try to delete the engine components
                    if hasattr(self.llm, 'llm_engine'):
                        rospy.loginfo("Deleting vLLM engine...")
                        engine = self.llm.llm_engine
                        
                        # Try to clean up engine components
                        if hasattr(engine, 'model_executor'):
                            try:
                                del engine.model_executor
                            except Exception as e:
                                rospy.logwarn(f"Error deleting model executor: {e}")
                        
                        if hasattr(engine, 'cache_engine'):
                            try:
                                del engine.cache_engine
                            except Exception as e:
                                rospy.logwarn(f"Error deleting cache engine: {e}")
                        
                        del self.llm.llm_engine
                    
                    # Delete the main LLM object
                    del self.llm
                    self.llm = None
                    
                    rospy.loginfo("vLLM object deleted successfully")
                    
                    # Wait a bit more for cleanup
                    time.sleep(3.0)
                    
                    # Force garbage collection multiple times
                    import gc
                    for i in range(3):
                        collected = gc.collect()
                        rospy.loginfo(f"Garbage collection pass {i+1}: collected {collected} objects")
                        time.sleep(1.0)
                    
                    # Try to clear CUDA cache multiple times with delays
                    try:
                        import torch
                        if torch.cuda.is_available():
                            rospy.loginfo("Clearing CUDA cache...")
                            for i in range(3):
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()  # Wait for all operations to complete
                                time.sleep(1.0)
                                rospy.loginfo(f"CUDA cache clear pass {i+1}")
                            
                            # Try to reset CUDA context if possible
                            try:
                                torch.cuda.reset_peak_memory_stats()
                                rospy.loginfo("Reset CUDA memory stats")
                            except Exception as e:
                                rospy.logwarn(f"Could not reset CUDA stats: {e}")
                                
                    except ImportError:
                        pass
                    except Exception as e:
                        rospy.logwarn(f"Error during CUDA cleanup: {e}")
                    
                    rospy.loginfo("Extended vLLM model cleanup completed")
                    
                except Exception as e:
                    rospy.logerr(f"Error during vLLM cleanup: {e}")
                    rospy.logwarn("Attempting force cleanup...")
                    
                    # Force cleanup - just delete everything we can
                    try:
                        if hasattr(self, 'llm'):
                            self.llm = None
                        import gc
                        gc.collect()
                        
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                            
                    except Exception as force_e:
                        rospy.logerr(f"Even force cleanup failed: {force_e}")
                    
        except Exception as e:
            rospy.logerr(f"Fatal error in vLLM cleanup: {e}")
        
        finally:
            # Final safety delay to let GPU operations complete
            rospy.loginfo("Final cleanup delay...")
            time.sleep(5.0)
            rospy.loginfo("vLLM cleanup procedure completed")

def check_system_requirements():
    """Check system requirements before starting the node."""
    try:
        # Check if vLLM is available
        try:
            import vllm
            rospy.loginfo(f"vLLM version: {vllm.__version__ if hasattr(vllm, '__version__') else 'unknown'}")
        except ImportError:
            rospy.logerr("vLLM is not installed. Please install vLLM first.")
            return False
        
        # Check PyTorch and CUDA
        try:
            import torch
            rospy.loginfo(f"PyTorch version: {torch.__version__}")
            
            if not torch.cuda.is_available():
                rospy.logerr("CUDA is not available. This node requires GPU support.")
                return False
                
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                rospy.logerr("No GPUs detected. This node requires at least one GPU.")
                return False
                
            rospy.loginfo(f"CUDA available with {gpu_count} GPU(s)")
            
        except ImportError:
            rospy.logerr("PyTorch is not installed.")
            return False
        
        return True
        
    except Exception as e:
        rospy.logerr(f"System requirements check failed: {e}")
        return False

def main():
    """Main function with proper multiprocessing setup and aggressive cleanup."""
    # Set multiprocessing start method for vLLM compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass
    
    node = None
    try:
        # Check system requirements first
        if not check_system_requirements():
            rospy.logerr("System requirements not met. Exiting.")
            sys.exit(1)
        
        rospy.loginfo("System requirements check passed. Initializing node...")
        node = QwenVLvLLMNode()
        node.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupted, shutting down...")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received, shutting down...")
    except ImportError as e:
        rospy.logerr(f"Import error - missing dependencies: {e}")
        rospy.logerr("Please ensure vLLM and all dependencies are properly installed")
    except RuntimeError as e:
        rospy.logerr(f"Runtime error: {e}")
        if "CUDA" in str(e):
            rospy.logerr("GPU/CUDA related error. Check GPU availability and memory.")
    except Exception as e:
        rospy.logerr(f"Failed to start QwenVL vLLM node: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Extended cleanup sequence
        rospy.loginfo("Starting extended cleanup sequence...")
        
        # Ensure cleanup happens even if node creation failed
        if node is not None:
            try:
                rospy.loginfo("Initiating node shutdown with extended timeout...")
                node.shutdown()
                
                # Additional wait time for cleanup to complete
                rospy.loginfo("Waiting additional time for complete cleanup...")
                time.sleep(10.0)
                
            except Exception as e:
                rospy.logerr(f"Error during final cleanup: {e}")
                # Force cleanup even if normal cleanup fails
                try:
                    node.cleanup()
                    time.sleep(5.0)
                except Exception as force_e:
                    rospy.logerr(f"Force cleanup also failed: {force_e}")
        
        # Final comprehensive cleanup
        try:
            rospy.loginfo("Performing final system cleanup...")
            
            # Multiple garbage collection passes
            import gc
            for i in range(5):
                collected = gc.collect()
                rospy.loginfo(f"Final GC pass {i+1}: collected {collected} objects")
                time.sleep(1.0)
            
            # Extended CUDA cleanup
            try:
                import torch
                if torch.cuda.is_available():
                    rospy.loginfo("Final CUDA cleanup...")
                    for i in range(5):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        time.sleep(1.0)
                    
                    # Try to reset CUDA context completely
                    try:
                        # This is aggressive but necessary for clean shutdown
                        torch.cuda.reset_peak_memory_stats()
                        rospy.loginfo("CUDA memory stats reset")
                    except Exception as cuda_e:
                        rospy.logwarn(f"CUDA reset failed: {cuda_e}")
                        
            except ImportError:
                pass
            except Exception as final_cuda_e:
                rospy.logerr(f"Final CUDA cleanup failed: {final_cuda_e}")
        
        except Exception as final_e:
            rospy.logerr(f"Final cleanup failed: {final_e}")
        
        finally:
            rospy.loginfo("QwenVL vLLM node shutdown sequence complete")
            rospy.loginfo("Waiting final delay before exit...")
            time.sleep(5.0)  # Final delay to ensure everything is cleaned up

if __name__ == '__main__':
    # This is CRITICAL for vLLM multiprocessing to work correctly
    # vLLM requires proper main module protection
    import multiprocessing
    
    # Force freeze support for multiprocessing
    try:
        multiprocessing.freeze_support()
    except:
        pass
    
    main()
