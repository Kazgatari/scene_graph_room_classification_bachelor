#!/usr/bin/env python3

# CRITICAL: Set environment variables and multiprocessing BEFORE any imports
import os
import multiprocessing
import sys
import signal
import atexit
import gc
import time
import threading
import traceback
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

# Set important environment variables first
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_INTERACTIVE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting

# Set multiprocessing method for compatibility
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

import numpy as np
import rospy
import torch
import json
import cv2
from PIL import Image as PILImage

# ROS imports
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs import point_cloud2 as pc2  # For PointCloud2 processing
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import message_filters

# Scene graph imports
from scene_graph.msg import (
    ObjectSceneGraph, ObjectDescription, ObjectAttribute, 
    ObjectInfo, ObjectPart, ObjectSpatialContext, DetectedObjects, DetectedObject
)

# Transformers imports
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor,
    GenerationConfig
)

class QwenVLTransformersNode:
    """
    High-performance QwenVL node using Transformers for object detection and scene graph generation.
    Features batch processing, optimized memory management, and comprehensive error handling.
    Mirrors the vLLM implementation architecture but uses Transformers backend.
    """
    
    def __init__(self):
        rospy.init_node('qwenvl_transformers_node', anonymous=True)
        
        # Initialize basic threading variables first (needed for cleanup)
        self.running = True
        self.shutdown_event = threading.Event()
        self.processing_queue = deque()
        self.queue_lock = threading.Lock()
        
        # Parameters (mirroring vLLM implementation)
        self.model_name = rospy.get_param('~model_name', 'Qwen/Qwen2-VL-7B-Instruct')  # Use standard model instead of AWQ
        
        # Fallback models in case of compatibility issues
        self.fallback_models = [
            'Qwen/Qwen2-VL-7B-Instruct',  # Older but more stable
            'Qwen/Qwen2.5-VL-2B-Instruct',  # Smaller fallback
            'Qwen/Qwen2.5-VL-7B-Instruct'
        ]
        self.detection_interval = rospy.get_param('~detection_interval', 1.0)
        self.max_objects = rospy.get_param('~max_objects', 50)
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.6)
        self.batch_size = rospy.get_param('~batch_size', 2)  # Smaller for Transformers
        self.max_queue_size = rospy.get_param('~max_queue_size', 16)
        self.gpu_memory_fraction = rospy.get_param('~gpu_memory_fraction', 0.7)
        
        # Initialize response storage for debugging
        self.model_responses = []
        self.response_count = 0
        self.debug_file_path = '/tmp/qwen_model_responses.txt'
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        self.last_detection_time = 0
        
        # Initialize model
        rospy.loginfo("Initializing Qwen2.5-VL Transformers model...")
        if not self.initialize_model_safe():
            rospy.logfatal("Model initialization failed - cannot continue")
            rospy.signal_shutdown("Model initialization failed")
            return
        
        # Threading and batch management
        self.processing_queue = deque(maxlen=self.max_queue_size)
        self.result_callbacks = {}
        self.request_id_counter = 0
        
        # Setup ROS communication
        self.setup_ros_communication()
        
        # Start batch processing thread
        self.processing_thread = threading.Thread(target=self.batch_processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Register signal handlers and cleanup functions
        self.setup_signal_handlers()
        
        rospy.loginfo("QwenVL Transformers node initialized successfully!")
        rospy.loginfo(f"Model: {self.model_name}")
        rospy.loginfo(f"Batch size: {self.batch_size}")
        rospy.loginfo(f"Device: {self.device}")

    def initialize_model_safe(self):
        """Initialize the Transformers model with comprehensive safety checks."""
        try:
            rospy.loginfo("Starting safe model initialization...")
            
            # Device selection with automatic fallback
            if torch.cuda.is_available():
                try:
                    # Test CUDA compatibility
                    test_tensor = torch.tensor([1.0]).cuda()
                    test_result = test_tensor + 1.0
                    self.device = "cuda"
                    self.torch_dtype = torch.float16
                    rospy.loginfo(f"CUDA available and compatible, using GPU")
                    rospy.loginfo(f"GPU: {torch.cuda.get_device_name(0)}")
                    rospy.loginfo(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                except Exception as e:
                    rospy.logwarn(f"CUDA available but incompatible ({e}), falling back to CPU")
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
            else:
                rospy.loginfo("CUDA not available, using CPU")
                self.device = "cpu"
                self.torch_dtype = torch.float32
            
            # Clear GPU memory before initialization
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Model loading with progressive fallback strategy
            model_loaded = False
            
            # Try different models and loading strategies
            for model_name in self.fallback_models:
                if model_loaded:
                    break
                    
                rospy.loginfo(f"Trying model: {model_name}")
                
                loading_strategies = [
                    # Strategy 1: Full optimization
                    {
                        "torch_dtype": self.torch_dtype,
                        "device_map": "auto" if self.device == "cuda" else None,
                        "trust_remote_code": True,
                        "attn_implementation": "sdpa",
                        "low_cpu_mem_usage": True,
                        "use_cache": True
                    },
                    # Strategy 2: No device map
                    {
                        "torch_dtype": self.torch_dtype,
                        "trust_remote_code": True,
                        "attn_implementation": "sdpa",
                        "low_cpu_mem_usage": True
                    },
                    # Strategy 3: Basic float32
                    {
                        "torch_dtype": torch.float32,
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True
                    },
                    # Strategy 4: Minimal config
                    {
                        "trust_remote_code": True
                    }
                ]
                
                for i, strategy in enumerate(loading_strategies):
                    try:
                        rospy.loginfo(f"  Trying loading strategy {i+1}/{len(loading_strategies)}")
                        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                            model_name,
                            **strategy
                        )
                        
                        # Move to device if needed
                        if strategy.get("device_map") is None:
                            self.model = self.model.to(self.device)
                        
                        # Update dtype if changed
                        if strategy.get("torch_dtype") == torch.float32:
                            self.torch_dtype = torch.float32
                        
                        # Set to evaluation mode
                        self.model.eval()
                        model_loaded = True
                        self.model_name = model_name  # Update to successful model
                        rospy.loginfo(f"✓ Model {model_name} loaded successfully with strategy {i+1}")
                        break
                        
                    except Exception as e:
                        rospy.logwarn(f"  Strategy {i+1} failed: {str(e)[:100]}...")
                        continue
            
            if not model_loaded:
                raise RuntimeError("Failed to load any compatible model")
            
            # Load processor (matching the successfully loaded model)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,  # This is now the successfully loaded model name
                trust_remote_code=True
            )
            
            rospy.loginfo("Model and processor loaded successfully!")
            
            # Setup optimized generation parameters (mirroring vLLM sampling params)
            self.setup_generation_config()
            
            # Test inference
            rospy.loginfo("Running test inference...")
            test_image = PILImage.new('RGB', (224, 224), color='red')
            test_prompt = "What is in this image?"
            
            success = self.test_inference(test_image, test_prompt)
            if not success:
                rospy.logerr("Test inference failed")
                return False
            
            rospy.loginfo("Test inference successful!")
            return True
            
        except Exception as e:
            rospy.logerr(f"Model initialization failed: {e}")
            traceback.print_exc()
            return False

    def test_inference(self, test_image, test_prompt):
        """Test model inference capability."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": test_prompt},
                    ],
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=50,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Fixed decoding logic - ensure proper token ID handling
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]  # Get only new tokens
            
            # Decode the generated tokens directly
            response_text = self.processor.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            if response_text and len(response_text.strip()) > 0:
                rospy.logdebug(f"Test response: {response_text[:100]}...")
                return True
            else:
                rospy.logwarn("Empty response from test inference")
                return False
            
        except Exception as e:
            rospy.logerr(f"Test inference error: {e}")
            traceback.print_exc()
            return False

    def setup_generation_config(self):
        """Setup optimized generation configuration (mirroring vLLM sampling params)."""
        try:
            # Get tokenizer pad and eos tokens safely
            pad_token_id = None
            eos_token_id = None
            
            if hasattr(self.processor, 'tokenizer'):
                pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id', None)
                eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id', None)
            
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.01,  # Very low for deterministic JSON output
                top_p=0.95,       # Slightly higher for better token diversity
                top_k=50,         # Add top_k for more focused sampling
                max_new_tokens=1024,   # Increased for more complete responses
                repetition_penalty=1.05,  # Light penalty to avoid repetition
                length_penalty=0.95,      # Slight bias toward shorter responses
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                return_dict_in_generate=False,  # Return simple tensor, not dict
                output_scores=False,
            )
            
            rospy.loginfo("Generation config setup complete")
            
        except Exception as e:
            rospy.logwarn(f"Error setting up generation config: {e}")
            # Use basic config as fallback
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.01,
                max_new_tokens=1024,
                use_cache=True,
                return_dict_in_generate=False,  # Return simple tensor
            )

    def setup_ros_communication(self):
        """Setup ROS subscribers and publishers (mirroring vLLM implementation)."""
        # Publishers (matching vLLM implementation)
        self.result_pub = rospy.Publisher(
            '/scene_graph/parser_llm/result', 
            ObjectSceneGraph, 
            queue_size=10
        )
        
        # Test individual subscribers first for debugging
        rospy.loginfo("Setting up individual subscribers for testing...")
        
        # Add simple test subscribers to verify topics are available
        def test_image_callback(msg):
            rospy.loginfo_once(f"✓ Received first image message: {msg.width}x{msg.height}")
            
        def test_depth_callback(msg):
            rospy.loginfo_once(f"✓ Received first depth message: frame={msg.header.frame_id}")
            
        def test_odom_callback(msg):
            rospy.loginfo_once(f"✓ Received first odom message: frame={msg.header.frame_id}")
        
        # Create test subscribers
        self.test_image_sub = rospy.Subscriber('/camera/color/image_raw', Image, test_image_callback)
        self.test_depth_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, test_depth_callback)
        self.test_odom_sub = rospy.Subscriber('/odom', Odometry, test_odom_callback)
        
        # Wait a moment for test subscribers to potentially receive messages
        rospy.sleep(1.0)
        
        # Option to use simple image subscriber if synchronizer doesn't work
        self.use_simple_callback = rospy.get_param('~use_simple_callback', False)
        
        if self.use_simple_callback:
            rospy.logwarn("Using simple image-only callback (synchronizer disabled)")
            self.simple_image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.simple_image_callback)
        else:
            # Subscribers with message filtering for synchronization
            self.image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
            self.depth_sub = message_filters.Subscriber('/camera/depth/points', PointCloud2)
            self.odom_sub = message_filters.Subscriber('/odom', Odometry)
            
            # Synchronize image and depth messages (using same settings as florence2_base.py)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub, self.odom_sub], 
                queue_size=100,  # Increased from 10 to match florence2_base.py 
                slop=0.1
            )
            self.ts.registerCallback(self.synchronized_callback)
        
        rospy.loginfo("ROS communication setup complete")
        rospy.loginfo("Subscribers created for:")
        rospy.loginfo("  - /camera/color/image_raw")
        rospy.loginfo("  - /camera/depth/points") 
        rospy.loginfo("  - /odom")
        rospy.loginfo("Waiting for synchronized messages...")

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            rospy.loginfo(f"Received signal {signal_name}, initiating graceful shutdown...")
            self.shutdown()
            # Give more time for cleanup, then force exit
            import threading
            def force_exit():
                time.sleep(8.0)  # Wait 8 seconds for cleanup (increased from 2)
                rospy.logwarn("Force terminating process...")
                os._exit(1)
            threading.Thread(target=force_exit, daemon=True).start()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        
        # Register atexit handler for unexpected exits
        atexit.register(self.cleanup_on_exit)

    def synchronized_callback(self, image_msg: Image, depth_msg: PointCloud2, odom_msg: Odometry):
        """Handle synchronized image and depth messages (mirroring vLLM implementation)."""
        try:
            rospy.loginfo("=== Synchronized callback triggered! ===")
            current_time = time.time()
            
            rospy.logdebug(f"Received synchronized messages:")
            rospy.logdebug(f"  Image: {image_msg.width}x{image_msg.height}, frame: {image_msg.header.frame_id}")
            rospy.logdebug(f"  Depth: frame: {depth_msg.header.frame_id}")
            rospy.logdebug(f"  Odom: frame: {odom_msg.header.frame_id}")
            
            # Rate limiting
            if current_time - self.last_detection_time < self.detection_interval:
                rospy.logdebug(f"Rate limiting: skipping (last: {current_time - self.last_detection_time:.2f}s ago)")
                return
            
            rospy.loginfo("Processing image for object detection...")
            
            # Convert ROS images to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.logdebug(f"Converted image to OpenCV: {cv_image.shape}")
            
            # For PointCloud2, we'll store it directly as it requires different processing
            # If you need to convert it to a depth image, additional processing is needed
            depth_data = depth_msg  # Store the PointCloud2 message directly
            
            # Add to processing queue with metadata
            request_data = {
                'request_id': self.request_id_counter,
                'timestamp': current_time,
                'image': cv_image,
                'depth': depth_data,  # PointCloud2 message
                'header': image_msg.header,
                'callback': self.publish_results
            }
            
            with self.queue_lock:
                if len(self.processing_queue) < self.max_queue_size:
                    self.processing_queue.append(request_data)
                    self.request_id_counter += 1
                    rospy.loginfo(f"Added image to processing queue (queue size: {len(self.processing_queue)})")
                else:
                    # Remove oldest if queue is full
                    self.processing_queue.popleft()
                    self.processing_queue.append(request_data)
                    self.request_id_counter += 1
                    rospy.logwarn(f"Queue full, replaced oldest item (queue size: {len(self.processing_queue)})")
            
            self.last_detection_time = current_time
            rospy.logdebug("Added image to processing queue")
                    
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")
            traceback.print_exc()

    def simple_image_callback(self, image_msg: Image):
        """Simple image-only callback for fallback when synchronizer doesn't work."""
        try:
            rospy.loginfo("=== Simple image callback triggered! ===")
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_detection_time < self.detection_interval:
                rospy.logdebug(f"Rate limiting: skipping (last: {current_time - self.last_detection_time:.2f}s ago)")
                return
            
            rospy.loginfo("Processing image for object detection (simple mode)...")
            
            # Convert ROS images to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.logdebug(f"Converted image to OpenCV: {cv_image.shape}")
            
            # Create fake depth and odom for compatibility
            fake_depth = PointCloud2()
            fake_depth.header = image_msg.header
            
            # Add to processing queue with metadata
            request_data = {
                'request_id': self.request_id_counter,
                'timestamp': current_time,
                'image': cv_image,
                'depth': fake_depth,  # Fake PointCloud2 message
                'header': image_msg.header,
                'callback': self.publish_results
            }
            
            with self.queue_lock:
                if len(self.processing_queue) < self.max_queue_size:
                    self.processing_queue.append(request_data)
                    self.request_id_counter += 1
                    rospy.loginfo(f"Added image to processing queue (simple mode, queue size: {len(self.processing_queue)})")
                else:
                    # Remove oldest if queue is full
                    self.processing_queue.popleft()
                    self.processing_queue.append(request_data)
                    self.request_id_counter += 1
                    rospy.logwarn(f"Queue full, replaced oldest item (simple mode, queue size: {len(self.processing_queue)})")
            
            self.last_detection_time = current_time
                    
        except Exception as e:
            rospy.logerr(f"Error in simple image callback: {e}")
            traceback.print_exc()

    def batch_processing_loop(self):
        """Main batch processing loop for optimal throughput (mirroring vLLM implementation)."""
        try:
            while self.running and not rospy.is_shutdown() and not self.shutdown_event.is_set():
                try:
                    # Check shutdown more frequently
                    if self.shutdown_event.is_set():
                        rospy.loginfo("Shutdown event detected in batch processing loop")
                        break
                    
                    # Collect batch
                    batch_requests = []
                    with self.queue_lock:
                        batch_size = min(self.batch_size, len(self.processing_queue))
                        for _ in range(batch_size):
                            if self.processing_queue:
                                batch_requests.append(self.processing_queue.popleft())
                    
                    if batch_requests:
                        self.process_batch(batch_requests)
                    else:
                        # Check shutdown event more frequently during idle time
                        for _ in range(10):  # Check 10 times over 0.1 seconds
                            if self.shutdown_event.is_set():
                                break
                            time.sleep(0.01)  # 10ms increments
                        
                except Exception as e:
                    rospy.logerr(f"Error in batch processing: {e}")
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(0.5)  # Shorter pause on error to be more responsive
        
        except Exception as e:
            if self.running:  # Only log if we're supposed to be running
                rospy.logerr(f"Fatal error in batch processing loop: {e}")
        finally:
            rospy.loginfo("Batch processing thread terminated")

    def process_batch(self, batch_requests: List[Dict]):
        """Process a batch of image requests using Transformers with true batching for better performance."""
        try:
            start_time = time.time()
            
            # Separate single image processing vs true batching based on batch size
            if len(batch_requests) == 1:
                # Single image - use optimized single processing
                self.process_single_request(batch_requests[0])
            else:
                # Multiple images - try true batch processing if beneficial
                if self.can_batch_efficiently(len(batch_requests)):
                    self.process_true_batch(batch_requests)
                else:
                    # Fall back to sequential processing for better reliability
                    for request in batch_requests:
                        if self.shutdown_event.is_set() or not self.running:
                            rospy.loginfo("Shutdown detected during batch processing, stopping...")
                            break
                        self.process_single_request(request)
            
            processing_time = time.time() - start_time
            rospy.loginfo(f"Processed batch of {len(batch_requests)} images in {processing_time:.3f}s")
            
        except Exception as e:
            rospy.logerr(f"Error in batch processing: {e}")
            traceback.print_exc()

    def can_batch_efficiently(self, batch_size):
        """Determine if true batching would be more efficient."""
        # True batching is only beneficial for larger batches and when using GPU
        # For vision-language models, batch processing can be memory intensive
        return (batch_size >= 3 and 
                hasattr(self, 'device') and 
                self.device == "cuda" and 
                batch_size <= 4)  # Limit to prevent OOM

    def process_single_request(self, request):
        """Process a single image request efficiently."""
        try:
            # Convert image to PIL format
            cv_image = request['image']
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Generate detection response
            detection_response = self.generate_detection_response(pil_image)
            
            if detection_response and 'objects' in detection_response:
                # Parse detection output 
                detected_objects = self.parse_detection_output(
                    json.dumps(detection_response), 
                    cv_image, 
                    request['depth']
                )
                
                # Publish results using callback
                if request['callback'] and detected_objects:
                    request['callback'](detected_objects, request['header'])
                
            else:
                rospy.logwarn("No objects detected in response")
                
        except Exception as e:
            rospy.logerr(f"Error processing single request: {e}")

    def process_true_batch(self, batch_requests: List[Dict]):
        """Process multiple images in a true batch for better GPU utilization."""
        try:
            rospy.loginfo(f"Processing true batch of {len(batch_requests)} images...")
            
            # Prepare batch data
            pil_images = []
            prompts = []
            
            prompt_template = self.create_detection_prompt()
            
            for request in batch_requests:
                # Convert image to PIL format
                cv_image = request['image']
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                pil_image = PILImage.fromarray(rgb_image)
                pil_images.append(pil_image)
                prompts.append(prompt_template)
            
            # Prepare batch messages
            batch_messages = []
            for pil_image, prompt in zip(pil_images, prompts):
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }]
                batch_messages.append(messages)
            
            # Process batch
            batch_texts = []
            batch_image_inputs = []
            
            for messages in batch_messages:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)
                
                image_inputs, _ = self.process_vision_info(messages)
                batch_image_inputs.extend(image_inputs)
            
            # Tokenize batch
            inputs = self.processor(
                text=batch_texts,
                images=batch_image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate batch response with simple settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.01,
                    max_new_tokens=1024,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            # This should return a simple tensor
            generated_ids = outputs
            
            # Decode batch responses
            input_length = inputs['input_ids'].shape[1]
            
            for i, request in enumerate(batch_requests):
                try:
                    # Extract tokens for this specific item in the batch
                    generated_tokens = generated_ids[i:i+1, input_length:]
                    
                    output_text = self.processor.tokenizer.decode(
                        generated_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    
                    # Parse and process results
                    detection_response = self.parse_json_response(output_text)
                    
                    if detection_response and 'objects' in detection_response:
                        detected_objects = self.parse_detection_output(
                            json.dumps(detection_response), 
                            request['image'], 
                            request['depth']
                        )
                        
                        if request['callback'] and detected_objects:
                            request['callback'](detected_objects, request['header'])
                    else:
                        rospy.logwarn(f"No objects detected in batch item {i}")
                        
                except Exception as e:
                    rospy.logerr(f"Error processing batch item {i}: {e}")
            
        except Exception as e:
            rospy.logerr(f"Error in true batch processing, falling back to sequential: {e}")
            # Fall back to sequential processing
            for request in batch_requests:
                if self.shutdown_event.is_set() or not self.running:
                    break
                self.process_single_request(request)

    def process_vision_info(self, messages):
        """Process vision information from messages (local implementation)."""
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content.get("type") == "image" and "image" in content:
                        image_inputs.append(content["image"])
                    elif content.get("type") == "video" and "video" in content:
                        video_inputs.append(content["video"])
        
        # Return None for videos if empty to avoid processing issues
        return image_inputs, video_inputs if video_inputs else None

    def generate_detection_response(self, pil_image):
        """Generate object detection response using Transformers (fixed decoding)."""
        try:
            # Create the prompt for object detection (same as vLLM implementation)
            prompt = self.create_detection_prompt()
            
            # Prepare the conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # Prepare inputs for the model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with simple settings (avoid complex GenerationConfig)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.1,  # Increased from 0.01 to allow more variation
                    max_new_tokens=1024,  # Increased from 256 to allow complete JSON
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.05,  # Prevent repetition
                    length_penalty=1.0,  # Neutral length preference
                )
            
            # This should return a simple tensor
            generated_ids = outputs
            
            # Fixed decoding logic - ensure proper token ID handling
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = generated_ids[:, input_length:]  # Get only new tokens
            
            # Decode the generated tokens directly
            output_text = self.processor.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Store response for debugging
            self.store_model_response(output_text)
            
            # Parse JSON response
            return self.parse_json_response(output_text)
            
        except Exception as e:
            rospy.logerr(f"Error generating detection response: {e}")
            traceback.print_exc()
            return None

    def create_detection_prompt(self) -> str:
        
        return """Analyze this image and detect all objects. You must respond with ONLY valid JSON in the exact format below, with no additional text, explanations, or markdown formatting.

CRITICAL: Your response must be valid JSON that can be parsed. Do not include any text before or after the JSON. Do not use "..." or truncate any values.

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

Rules:
- Unique object detection is required (dont repeat objects with identical bounding boxes)
- Maximum 8 unique objects can be detected
- Bounding box coordinates are percentile values from 0 to 999 (0 = top/left, 999 = bottom/right)
- Use complete attribute values, never "..." or truncation
- Include empty string "" for unknown attributes
- Relations describe spatial relationships: "on", "under", "attached_to", "in"
- Response must be parseable JSON only

Detect all objects in the image and respond with the JSON:"""

    def parse_json_response(self, response_text):
        """Parse JSON response from model output with comprehensive error handling."""
        if not response_text or not response_text.strip():
            rospy.logwarn("Empty response from model")
            return None
            
        try:
            # Try direct JSON parsing first
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            rospy.logdebug(f"Direct JSON parse failed: {e}")
            
            # Clean the response text
            import re
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```\s*$', '', cleaned_text)
            cleaned_text = re.sub(r'^```\s*', '', cleaned_text)
            
            # Remove any leading/trailing non-JSON text
            cleaned_text = cleaned_text.strip()
            
            # Try parsing the cleaned text
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e2:
                rospy.logdebug(f"Cleaned JSON parse failed: {e2}")
                
                # Extract JSON object from the text (more robust pattern)
                json_patterns = [
                    r'\{[^{}]*"objects"[^{}]*\[.*?\]\s*\}',  # Simple objects array
                    r'\{.*?"objects"\s*:\s*\[.*?\].*?\}',     # Objects with other fields
                    r'\{.*?\}',                               # Any JSON object
                ]
                
                for pattern in json_patterns:
                    json_match = re.search(pattern, cleaned_text, re.DOTALL)
                    if json_match:
                        try:
                            json_str = json_match.group()
                            # Fix common JSON issues
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            continue
                
                # Log the problematic response for debugging
                rospy.logwarn(f"Failed to parse JSON. Response preview: {response_text[:300]}...")
                rospy.logdebug(f"Full response: {response_text}")
                
                # Store the failed response for debugging
                self.store_model_response(response_text, status="parse_failed", 
                                        error_message=f"JSON parsing failed: {e}")
                
                return None

    def parse_detection_output(
        self, 
        output_text: str, 
        image: np.ndarray, 
        depth_msg: PointCloud2
    ) -> List[Dict[str, Any]]:
        """Parse the model output and extract object detections (adapted from vLLM implementation)."""
        try:
            # Parse JSON response
            json_data = json.loads(output_text)
            
            if 'objects' in json_data:
                detections = json_data['objects']
            else:
                # Assume the entire response is the objects array
                detections = json_data if isinstance(json_data, list) else [json_data]
            
            # Process each detection
            processed_objects = []
            height, width = image.shape[:2]
            
            # Note: PointCloud2 processing would require additional conversion
            # if depth information is needed for object localization
            
            for detection in detections:
                try:
                    bbox = detection.get('bounding_box', [0, 0, 100, 100])
                    
                    # Normalize bounding box if needed
                    if len(bbox) >= 4:
                        x_min, y_min, x_max, y_max = bbox[:4]
                        
                        # Convert from percentage (0-999) to pixel coordinates
                        if max(bbox) > 1.0:  # Assume percentage format
                            x_min = (x_min / 999.0) * width
                            y_min = (y_min / 999.0) * height
                            x_max = (x_max / 999.0) * width
                            y_max = (y_max / 999.0) * height
                    
                    processed_obj = {
                        'label': detection.get('label', 'unknown'),
                        'bounding_box': [x_min, y_min, x_max, y_max],
                        'attributes': detection.get('attributes', {}),
                        'relations': detection.get('relations', []),
                        'confidence': self.confidence_threshold
                    }
                    
                    processed_objects.append(processed_obj)
                    
                except Exception as e:
                    rospy.logwarn(f"Error processing detection: {e}")
                    continue
            
            rospy.loginfo(f"Detected {len(processed_objects)} objects")
            return processed_objects
            
        except json.JSONDecodeError as e:
            rospy.logerr(f"JSON parsing error: {e}")
            rospy.logdebug(f"Failed to parse JSON: {output_text[:500]}...")
            return []
        except Exception as e:
            rospy.logerr(f"Error parsing detection output: {e}")
            return []

    def publish_results(self, detected_objects: List[Dict], header):
        """Publish detected objects as ObjectSceneGraph message (fixed header issue)."""
        try:
            if not detected_objects:
                rospy.logwarn("No objects detected to publish")
                return
            
            # Create ObjectSceneGraph message for the most prominent object
            main_object_data = detected_objects[0]
            
            msg = ObjectSceneGraph()
            # Note: ObjectSceneGraph doesn't have a header attribute, skipping header assignment
            
            # Main object
            msg.main_object.id = 0
            msg.main_object.name = main_object_data['label']
            
            # Attributes for main object
            attributes = main_object_data.get('attributes', {})
            msg.main_object.attributes.color = attributes.get('color', '')
            # material removed from ObjectAttribute
            msg.main_object.attributes.style = attributes.get('style', '')
            
            # Parts - treat other detected objects as parts if they have relations to the main object
            relations = main_object_data.get('relations', [])
            main_obj_name = main_object_data['label']
            
            for obj_data in detected_objects[1:]:  # Skip the main object
                obj_relations = obj_data.get('relations', [])
                
                # Check if this object is related to the main object
                is_part = any(
                    rel.get('target', '').lower() == main_obj_name.lower() and 
                    rel.get('type') in ['part_of', 'attached_to']
                    for rel in obj_relations
                )
                
                if is_part:
                    # Add as part
                    part = ObjectPart()
                    part.name = obj_data['label']
                    part.relationship_to_main = 'part_of'
                    
                    part_attrs = obj_data.get('attributes', {})
                    part.attributes.color = part_attrs.get('color', '')
                    # material removed
                    part.attributes.style = part_attrs.get('style', '')
                    
                    msg.parts.append(part)
                else:
                    # Add as environment object
                    env_obj = ObjectInfo()
                    env_obj.id = len(msg.environment) + 1
                    env_obj.name = obj_data['label']
                    
                    env_attrs = obj_data.get('attributes', {})
                    env_obj.attributes.color = env_attrs.get('color', '')
                    # material removed
                    env_obj.attributes.style = env_attrs.get('style', '')
                    
                    msg.environment.append(env_obj)
            
            # Spatial context - extract spatial relationships from all objects
            nearby_objects = []
            spatial_relations = []
            
            for obj_data in detected_objects:
                for rel in obj_data.get('relations', []):
                    if rel.get('type') in ['near', 'on', 'under', 'in']:
                        target = rel.get('target', '')
                        if target:
                            nearby_objects.append(target)
                            spatial_relations.append(f"{obj_data['label']} {rel.get('type')} {target}")
            
            msg.spatial_context.position = "; ".join(spatial_relations) if spatial_relations else ""
            msg.spatial_context.nearby_objects = list(set(nearby_objects))
            
            # Publish the message
            self.result_pub.publish(msg)
            
            rospy.loginfo(f"Published ObjectSceneGraph with main object: {msg.main_object.name}, "
                         f"{len(msg.parts)} parts, {len(msg.environment)} environment objects")
                
        except Exception as e:
            rospy.logerr(f"Error publishing results: {e}")
            traceback.print_exc()

    def run(self):
        """Main run loop."""
        rospy.loginfo("QwenVL Transformers node is running...")
        
        try:
            # Use rospy.spin() but catch interruption
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Received KeyboardInterrupt, shutting down...")
        except Exception as e:
            rospy.logerr(f"Unexpected error in main loop: {e}")
        finally:
            self.shutdown()
            # Force exit after shutdown
            rospy.loginfo("Forcing process termination...")
            os._exit(0)

    def store_model_response(self, raw_response, status="generated", error_message=None):
        """Store a model response for debugging purposes."""
        try:
            from datetime import datetime
            
            self.response_count += 1
            
            # Try to parse the response to see if it's valid
            parsed_successfully = False
            try:
                json.loads(raw_response)
                parsed_successfully = True
            except:
                pass
            
            response_data = {
                'id': self.response_count,
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'raw_response': raw_response,
                'parsed_successfully': parsed_successfully,
                'error_message': error_message,
                'response_length': len(raw_response) if raw_response else 0
            }
            
            self.model_responses.append(response_data)
            
            # Log preview of the response
            preview = raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
            rospy.loginfo(f"Stored response #{self.response_count}: parsed={parsed_successfully}, len={len(raw_response)}")
            rospy.logdebug(f"Response preview: {preview}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to store model response: {e}")

    def save_model_responses(self):
        """Save all model responses to a file for debugging."""
        try:
            if not self.model_responses:
                rospy.loginfo("No model responses to save")
                return
                
            import json
            from datetime import datetime
            
            # Create debug data structure
            debug_data = {
                'timestamp': datetime.now().isoformat(),
                'total_responses': len(self.model_responses),
                'responses': self.model_responses
            }
            
            # Save as JSON for better readability
            with open(self.debug_file_path, 'w') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
                
            # Also save as plain text for easy reading
            text_file_path = '/tmp/qwen_model_responses_readable.txt'
            with open(text_file_path, 'w') as f:
                f.write(f"Qwen Model Responses Debug Log\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Total responses: {len(self.model_responses)}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, response_data in enumerate(self.model_responses, 1):
                    f.write(f"Response #{i}\n")
                    f.write(f"Timestamp: {response_data.get('timestamp', 'Unknown')}\n")
                    f.write(f"Status: {response_data.get('status', 'Unknown')}\n")
                    f.write(f"Raw Response Length: {len(response_data.get('raw_response', ''))}\n")
                    f.write(f"Parsed Successfully: {response_data.get('parsed_successfully', False)}\n")
                    f.write("-" * 40 + "\n")
                    f.write("Raw Response:\n")
                    f.write(response_data.get('raw_response', 'No response'))
                    f.write("\n" + "-" * 40 + "\n")
                    if response_data.get('error_message'):
                        f.write(f"Error: {response_data['error_message']}\n")
                        f.write("-" * 40 + "\n")
                    f.write("\n")
                    
            rospy.loginfo(f"Model responses saved to:")
            rospy.loginfo(f"  JSON format: {self.debug_file_path}")
            rospy.loginfo(f"  Text format: {text_file_path}")
            
        except Exception as e:
            rospy.logerr(f"Failed to save model responses: {e}")

    def shutdown(self):
        """Graceful shutdown of the node (mirroring vLLM implementation)."""
        rospy.loginfo("Initiating graceful shutdown...")
        
        try:
            # Set shutdown flags first
            self.running = False
            self.shutdown_event.set()
            
            # Stop ROS subscribers to prevent new data
            try:
                # Clean up test subscribers
                if hasattr(self, 'test_image_sub'):
                    self.test_image_sub.unregister()
                if hasattr(self, 'test_depth_sub'):
                    self.test_depth_sub.unregister()
                if hasattr(self, 'test_odom_sub'):
                    self.test_odom_sub.unregister()
                
                # Clean up main subscribers
                if hasattr(self, 'image_sub'):
                    self.image_sub.unregister()
                if hasattr(self, 'depth_sub'):
                    self.depth_sub.unregister()
                if hasattr(self, 'odom_sub'):
                    self.odom_sub.unregister()
                if hasattr(self, 'simple_image_sub'):
                    self.simple_image_sub.unregister()
                if hasattr(self, 'ts'):
                    del self.ts
                rospy.loginfo("Stopped ROS subscribers")
            except Exception as e:
                rospy.logwarn(f"Error stopping subscribers: {e}")
            
            # Wait for processing thread to finish
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                rospy.loginfo("Waiting for processing thread to finish...")
                self.processing_thread.join(timeout=6.0)  # Increased timeout from 3 to 6 seconds
                if self.processing_thread.is_alive():
                    rospy.logwarn("Processing thread did not terminate gracefully")
                    # Try to force thread termination more aggressively
                    rospy.loginfo("Attempting to force thread termination...")
                    self.shutdown_event.set()  # Make sure shutdown event is set
                    # Give it one more chance
                    self.processing_thread.join(timeout=1.0)
                    if self.processing_thread.is_alive():
                        rospy.logwarn("Thread still alive after forced termination attempt")
            
            # Save model responses for debugging
            self.save_model_responses()
            
            # Clean up model resources
            self.cleanup_model()
            
            rospy.loginfo("Graceful shutdown completed")
            
        except Exception as e:
            rospy.logerr(f"Error during shutdown: {e}")
            traceback.print_exc()
        finally:
            # Force ROS shutdown
            try:
                rospy.signal_shutdown("Node shutdown completed")
            except:
                pass

    def cleanup_on_exit(self):
        """Cleanup function called on unexpected exit."""
        try:
            if hasattr(self, 'running') and self.running:
                rospy.loginfo("Emergency cleanup initiated...")
                self.cleanup_model()
                # Force exit after emergency cleanup
                os._exit(1)
        except Exception as e:
            print(f"Error during emergency cleanup: {e}")
            os._exit(1)

    def cleanup_model(self):
        """Clean up model resources."""
        cleanup_start_time = time.time()
        max_cleanup_time = 3.0  # Maximum time to spend on cleanup
        
        try:
            rospy.loginfo("Cleaning up model resources...")
            
            # Clean up model
            if hasattr(self, 'model') and self.model is not None:
                try:
                    if time.time() - cleanup_start_time > max_cleanup_time:
                        rospy.logwarn("Cleanup timeout reached, skipping model deletion")
                        return
                    del self.model
                    self.model = None
                    rospy.logdebug("Model deleted")
                except Exception as e:
                    rospy.logwarn(f"Error deleting model: {e}")
            
            # Clean up processor
            if hasattr(self, 'processor') and self.processor is not None:
                try:
                    if time.time() - cleanup_start_time > max_cleanup_time:
                        rospy.logwarn("Cleanup timeout reached, skipping processor deletion")
                        return
                    del self.processor
                    self.processor = None
                    rospy.logdebug("Processor deleted")
                except Exception as e:
                    rospy.logwarn(f"Error deleting processor: {e}")
            
            # Clear CUDA cache if using GPU (with timeout protection)
            if hasattr(self, 'device') and self.device == "cuda":
                try:
                    if time.time() - cleanup_start_time > max_cleanup_time:
                        rospy.logwarn("Cleanup timeout reached, skipping CUDA cache clear")
                        return
                    rospy.logdebug("Clearing CUDA cache...")
                    torch.cuda.empty_cache()
                    # Skip synchronize as it might hang
                    # torch.cuda.synchronize()
                    rospy.logdebug("CUDA cache cleared")
                except Exception as e:
                    rospy.logwarn(f"Error clearing CUDA cache: {e}")
            
            # Force garbage collection
            try:
                if time.time() - cleanup_start_time > max_cleanup_time:
                    rospy.logwarn("Cleanup timeout reached, skipping garbage collection")
                    return
                gc.collect()
                rospy.logdebug("Garbage collection completed")
            except Exception as e:
                rospy.logwarn(f"Error during garbage collection: {e}")
            
            cleanup_time = time.time() - cleanup_start_time
            rospy.loginfo(f"Model cleanup completed in {cleanup_time:.2f}s")
            
        except Exception as e:
            rospy.logerr(f"Error during model cleanup: {e}")
            # Don't let cleanup errors prevent shutdown


def check_system_requirements():
    """Check system requirements before starting the node."""
    try:
        # Check PyTorch
        try:
            import torch
            rospy.loginfo(f"PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                rospy.loginfo(f"CUDA available with {gpu_count} GPU(s)")
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    rospy.loginfo(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
            else:
                rospy.logwarn("CUDA not available, will use CPU (slower)")
            
        except ImportError:
            rospy.logerr("PyTorch is not installed.")
            return False
        
        # Check Transformers
        try:
            import transformers
            rospy.loginfo(f"Transformers version: {transformers.__version__}")
        except ImportError:
            rospy.logerr("Transformers is not installed. Please install: pip install transformers")
            return False
        
        return True
        
    except Exception as e:
        rospy.logerr(f"System requirements check failed: {e}")
        return False


def main():
    """Main function with proper error handling and cleanup."""
    # Multiprocessing start method is already set at module level
    
    node = None
    try:
        # Check system requirements first
        if not check_system_requirements():
            rospy.logerr("System requirements not met. Exiting.")
            sys.exit(1)
        
        rospy.loginfo("System requirements check passed. Initializing node...")
        node = QwenVLTransformersNode()
        node.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupted, shutting down...")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt received, shutting down...")
    except ImportError as e:
        rospy.logerr(f"Import error - missing dependencies: {e}")
        rospy.logerr("Please ensure Transformers and all dependencies are properly installed")
    except RuntimeError as e:
        rospy.logerr(f"Runtime error: {e}")
        if "CUDA" in str(e):
            rospy.logerr("GPU/CUDA related error. Check GPU availability and memory.")
    except Exception as e:
        rospy.logerr(f"Failed to start QwenVL Transformers node: {e}")
        traceback.print_exc()
    finally:
        # Final cleanup only if node wasn't already cleaned up
        if node is not None and hasattr(node, 'running') and node.running:
            try:
                rospy.loginfo("Final cleanup in main...")
                node.cleanup_model()
            except Exception as e:
                rospy.logerr(f"Error during final cleanup: {e}")
        
        rospy.loginfo("QwenVL Transformers node shutdown complete")
        # Force process termination with proper cleanup
        try:
            rospy.signal_shutdown("Main function completed")
        except:
            pass
        
        # Clean up multiprocessing resources
        try:
            # Force cleanup of any remaining multiprocessing resources
            import multiprocessing.util
            multiprocessing.util._exit_function()
        except:
            pass
        
        os._exit(0)


if __name__ == '__main__':
    main()
