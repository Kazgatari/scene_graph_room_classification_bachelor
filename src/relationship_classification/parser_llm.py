#!/usr/bin/env python3
import os
# Set environment variables to reduce warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Set correct CUDA architecture for RTX 5070 Ti (Blackwell, compute capability 9.0) ==> for flashinfer
# os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

import json
import time
import torch
import threading
import pickle
import glob
from collections import deque
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from scene_graph.msg import ObjectSceneGraph, ObjectDescriptions, ObjectDescription, ObjectAttribute, ObjectInfo, ObjectPart, ObjectSpatialContext
from std_srvs.srv import Empty, EmptyResponse
import sys
import signal
import rospy

class QwenChatNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('parser_llm_node', anonymous=True)
        rospy.loginfo("Initializing LLM Parser Node...")
        
        # Console parameters for dumping and loading
        self.dump_mode = rospy.get_param('~dump_scene_graphs', False)
        self.load_mode = rospy.get_param('~load_scene_graphs', False)
        self.auto_publish_loaded = rospy.get_param('~auto_publish_loaded', True)  # Auto-publish loaded graphs
        self.save_loaded_as_txt = rospy.get_param('~save_loaded_as_txt', False)  # Save loaded graphs as txt file
        self.dump_directory = rospy.get_param('~dump_directory', '/root/catkin_ws/src/scene_graph_room_classification_bachelor/data/scene_graphs')
        self.txt_output_directory = rospy.get_param('~txt_output_directory', '/root/catkin_ws/src/scene_graph_room_classification_bachelor/scene_graph_objects')
        
        # Ensure dump directory exists
        if self.dump_mode and not os.path.exists(self.dump_directory):
            os.makedirs(self.dump_directory, exist_ok=True)
            rospy.loginfo(f"Created dump directory: {self.dump_directory}")
        
        # Ensure txt output directory exists if needed
        if self.save_loaded_as_txt and not os.path.exists(self.txt_output_directory):
            os.makedirs(self.txt_output_directory, exist_ok=True)
            rospy.loginfo(f"Created txt output directory: {self.txt_output_directory}")
        
        # Check for conflicting modes
        if self.dump_mode and self.load_mode:
            rospy.logerr("Cannot enable both dump_mode and load_mode simultaneously!")
            sys.exit(1)
        
        if self.load_mode:
            rospy.loginfo(f"LOAD MODE: Will replay ObjectSceneGraph from {self.dump_directory}")
            self.loaded_scene_graphs = self.load_all_scene_graphs()
            if not self.loaded_scene_graphs:
                rospy.logwarn("No saved ObjectSceneGraph files found in load mode!")
            else:
                if self.save_loaded_as_txt:
                    rospy.loginfo(f"Save as TXT enabled: Will save {len(self.loaded_scene_graphs)} loaded scene graphs to txt file")
                    self.save_loaded_scene_graphs_as_txt()
                if self.auto_publish_loaded:
                    rospy.loginfo(f"Auto-publish enabled: Will publish {len(self.loaded_scene_graphs)} loaded scene graphs after setup")
        elif self.dump_mode:
            rospy.loginfo(f"DUMP MODE: Will save ObjectSceneGraph to {self.dump_directory}")
        else:
            rospy.loginfo("NORMAL MODE: Standard LLM parsing")
        
        # Batch processing parameters
        self.batch_size = rospy.get_param('~batch_size', 10)  # Process in batches of 10
        self.batch_timeout = rospy.get_param('~batch_timeout', 2.0)  # Wait 2 seconds before processing partial batch
        self.request_queue = deque()
        self.queue_lock = threading.Lock()
        self.last_request_time = None
        
        # Model configuration
        self.model_name = "Qwen/Qwen3-4B-AWQ"
        #"Qwen/Qwen3-1.7B-GPTQ-Int8"
        #"Qwen/Qwen3-4B-AWQ"
        #"Qwen/Qwen3-4B-Instruct-2507"
        #"Qwen/Qwen3-8B"
        #"Qwen/Qwen2.5-1.5B-Instruct"
        
        # Only initialize LLM if not in load mode
        if not self.load_mode:
            rospy.loginfo("Loading Qwen model with vLLM... This may take a moment.")
            self.initialize_llm_model()
        else:
            rospy.loginfo("Skipping LLM initialization in load mode")
            self.llm = None
            self.tokenizer = None
        
        # Setup ROS subscribers and publishers
        self.setup_ros_communication()
        
        # Start batch processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self.batch_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        rospy.loginfo("LLM Parser Node initialized and ready!")

    def initialize_llm_model(self):
        """Initialize the LLM model and tokenizer"""
        try:
            # Load tokenizer for prompt formatting
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Ensure a pad token exists for generation
            if getattr(self.tokenizer, 'pad_token', None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Initialize vLLM model
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                rospy.loginfo("Using GPU for vLLM inference")
                self.llm = LLM(
                    model=self.model_name,
                    dtype="float16",
                    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
                    trust_remote_code=True,
                    tensor_parallel_size=1,      # Single GPU
                    max_model_len=4096,          # Context window
                    enforce_eager=False,         # Use CUDA graphs for speed
                    disable_log_stats=True       # Reduce logging overhead
                )
            else:
                rospy.loginfo("Using CPU for vLLM inference")
                self.llm = LLM(
                    model=self.model_name,
                    dtype="float32",
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    max_model_len=4096
                )

            # Set up sampling parameters for different use cases
            self.sampling_params_first = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
            )
            
            self.sampling_params_second = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=512,
                stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
            )

            rospy.loginfo("vLLM model loaded successfully!")
            rospy.loginfo(f"Batch size: {self.batch_size}, Batch timeout: {self.batch_timeout}s")
            rospy.loginfo(f"Using GPU: {use_gpu}")

        except Exception as e:
            rospy.logerr(f"Failed to load model '{self.model_name}': {e}")
            rospy.logerr("Make sure vLLM is properly installed: pip install vllm")
            sys.exit(1)


        self.system_prompt = """You are a scene description parser that extracts structured information from object descriptions for 3D scene graph construction.

Your task is to analyze text descriptions and extract:
1. MAIN_OBJECT: The primary object mentioned (usually first) with its attributes
2. PARTS: Components or parts of the main object that should be separate entities
3. ENVIRONMENT: Environmental elements like walls, floor, ceiling, room features
4. SPATIAL_CONTEXT: Any spatial relationships or positioning information

Output format must be valid JSON with this exact structure:

{
  "main_object": {
    "name": "object_name",
    "attributes": {
      "color": "color_value",
      "material": "material_value",
      "style": "style_value"
    }
  },
  "parts": [
    {
      "name": "part_name",
      "attributes": {
        "color": "color_value",
        "material": "material_value",
        "style": "style_value"
      },
      "relationship_to_main": "attached_to|part_of|component_of"
    }
  ],
  "environment": [
    {
      "name": "env_element_name",
      "attributes": {
        "color": "color_value",
        "material": "material_value",
        "style": "style_value"
      }
    }
  ],
  "spatial_context": {
    "position": "position_description",
    "nearby_objects": ["object1", "object2"]
  }
}

Rules:
- Only include attributes that are explicitly mentioned
- If no parts/environment/spatial context exists, use empty arrays/objects
- If an attribute is not mentioned, use empty strings
- Use consistent naming (lowercase with underscores)
- Colors should be specific when mentioned
- Materials should be extracted when mentioned (wood, metal, fabric, etc.)
- The main object is typically the first noun phrase in the description

Please provide only the JSON response without additional text or explanations."""

        self.running = True  # Flag to control the main loop

    def setup_ros_communication(self):
        """Setup ROS subscribers and publishers"""
        # Subscribe to incoming object descriptions
        self.description_sub = rospy.Subscriber(
            '/scene_graph/parser_llm', 
            ObjectDescriptions, 
            self.description_callback,
            queue_size=50
        )
        
        # Publisher for processed results
        self.result_pub = rospy.Publisher(
            '/scene_graph/parser_llm/result', 
            ObjectSceneGraph, 
            queue_size=10
        )
        
        # Service to manually trigger replay of loaded scene graphs
        if self.load_mode:
            self.replay_service = rospy.Service(
                '/scene_graph/parser_llm/replay_loaded', 
                Empty, 
                self.replay_service_callback
            )
            rospy.loginfo("Replay service available at /scene_graph/parser_llm/replay_loaded")
        
        rospy.loginfo("ROS communication setup complete")
        
        # Auto-publish loaded scene graphs if enabled
        if self.load_mode and self.auto_publish_loaded and self.loaded_scene_graphs:
            rospy.loginfo(f"Auto-publishing {len(self.loaded_scene_graphs)} loaded scene graphs...")
            self.publish_all_loaded_scene_graphs()

    def replay_service_callback(self, req):
        """Service callback to manually trigger replay of all loaded scene graphs"""
        if not self.load_mode:
            rospy.logwarn("Replay service called but not in load mode")
            return EmptyResponse()
        
        if not self.loaded_scene_graphs:
            rospy.logwarn("No loaded scene graphs to replay")
            return EmptyResponse()
        
        rospy.loginfo(f"Manual replay triggered via service: publishing {len(self.loaded_scene_graphs)} scene graphs")
        self.publish_all_loaded_scene_graphs()
        return EmptyResponse()

    def publish_all_loaded_scene_graphs(self):
        """Publish all loaded scene graphs with sequential object IDs"""
        if not self.loaded_scene_graphs:
            rospy.logwarn("No loaded scene graphs to publish")
            return
        
        published_count = 0
        for i, saved_data in enumerate(self.loaded_scene_graphs):
            try:
                # Use sequential object IDs starting from 1
                object_id = i + 1
                
                # Convert saved data back to ROS message
                scene_graph_msg = self.dict_to_scene_graph_msg(saved_data, object_id)
                
                # Publish the message
                self.result_pub.publish(scene_graph_msg)
                published_count += 1
                
                rospy.loginfo(f"Published loaded ObjectSceneGraph {i+1}/{len(self.loaded_scene_graphs)} (object_id: {object_id})")
                
                # Small delay to avoid overwhelming the system
                rospy.sleep(0.1)
                
            except Exception as e:
                rospy.logerr(f"Error publishing loaded scene graph {i}: {e}")
        
        rospy.loginfo(f"Successfully published {published_count}/{len(self.loaded_scene_graphs)} loaded scene graphs")

    def description_callback(self, msg: ObjectDescriptions):
        """Handle incoming ObjectDescriptions messages"""
        if self.load_mode:
            # In load mode, replay saved ObjectSceneGraph instead of processing
            self.replay_saved_scene_graphs(msg)
            return
        
        with self.queue_lock:
            for description in msg.objects:
                # Add each description to the processing queue
                self.request_queue.append({
                    'object_id': description.object_id,
                    'caption': description.caption,
                    'timestamp': time.time()
                })
                rospy.logdebug(f"Added object {description.object_id} to processing queue")
            
            self.last_request_time = time.time()
            rospy.loginfo(f"Received {len(msg.objects)} descriptions, queue size: {len(self.request_queue)}")

    def save_scene_graph(self, scene_graph_msg, object_id):
        """Save ObjectSceneGraph to disk"""
        if not self.dump_mode:
            return
        
        try:
            timestamp = int(time.time() * 1000)  # Milliseconds
            filename = f"scene_graph_obj_{object_id}_{timestamp}.pkl"
            filepath = os.path.join(self.dump_directory, filename)
            
            # Convert ROS message to dictionary for better serialization
            scene_graph_data = {
                'timestamp': timestamp,
                'object_id': object_id,
                'main_object': {
                    'id': scene_graph_msg.main_object.id,
                    'name': scene_graph_msg.main_object.name,
                    'attributes': {
                        'color': scene_graph_msg.main_object.attributes.color,
                        'material': scene_graph_msg.main_object.attributes.material,
                        'style': scene_graph_msg.main_object.attributes.style
                    }
                },
                'parts': [],
                'environment': [],
                'spatial_context': {
                    'position': scene_graph_msg.spatial_context.position,
                    'nearby_objects': list(scene_graph_msg.spatial_context.nearby_objects)
                }
            }
            
            # Add parts
            for part in scene_graph_msg.parts:
                part_data = {
                    'name': part.name,
                    'relationship_to_main': part.relationship_to_main,
                    'attributes': {
                        'color': part.attributes.color,
                        'material': part.attributes.material,
                        'style': part.attributes.style
                    }
                }
                scene_graph_data['parts'].append(part_data)
            
            # Add environment
            for env in scene_graph_msg.environment:
                env_data = {
                    'id': env.id,
                    'name': env.name,
                    'attributes': {
                        'color': env.attributes.color,
                        'material': env.attributes.material,
                        'style': env.attributes.style
                    }
                }
                scene_graph_data['environment'].append(env_data)
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(scene_graph_data, f)
            
            rospy.loginfo(f"Saved ObjectSceneGraph for object {object_id} to {filename}")
            
        except Exception as e:
            rospy.logerr(f"Error saving ObjectSceneGraph for object {object_id}: {e}")

    def load_all_scene_graphs(self):
        """Load all saved ObjectSceneGraph files"""
        if not os.path.exists(self.dump_directory):
            rospy.logwarn(f"Dump directory does not exist: {self.dump_directory}")
            return []
        
        try:
            pattern = os.path.join(self.dump_directory, "scene_graph_obj_*.pkl")
            files = glob.glob(pattern)
            
            loaded_graphs = []
            for filepath in sorted(files):
                try:
                    with open(filepath, 'rb') as f:
                        scene_graph_data = pickle.load(f)
                        loaded_graphs.append(scene_graph_data)
                except Exception as e:
                    rospy.logwarn(f"Error loading {filepath}: {e}")
            
            rospy.loginfo(f"Loaded {len(loaded_graphs)} ObjectSceneGraph files from {self.dump_directory}")
            return loaded_graphs
            
        except Exception as e:
            rospy.logerr(f"Error loading ObjectSceneGraph files: {e}")
            return []

    def save_loaded_scene_graphs_as_txt(self):
        """Save all loaded scene graphs to a text file for human-readable format"""
        if not self.load_mode or not self.save_loaded_as_txt:
            return
        
        if not self.loaded_scene_graphs:
            rospy.logwarn("No loaded scene graphs to save as txt")
            return
        
        try:
            # Create timestamped filename
            timestamp = int(time.time())
            filename = f"loaded_scene_graphs_{timestamp}.txt"
            filepath = os.path.join(self.txt_output_directory, filename)
            
            rospy.loginfo(f"Saving {len(self.loaded_scene_graphs)} loaded scene graphs to {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"LOADED SCENE GRAPHS - {len(self.loaded_scene_graphs)} objects\n")
                f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}\n")
                f.write(f"Source directory: {self.dump_directory}\n")
                f.write("=" * 80 + "\n\n")
                
                for i, scene_graph_data in enumerate(self.loaded_scene_graphs):
                    f.write(f"OBJECT {i+1}/{len(self.loaded_scene_graphs)}\n")
                    f.write("-" * 40 + "\n")
                    
                    # Write metadata
                    f.write(f"Original Object ID: {scene_graph_data.get('object_id', 'N/A')}\n")
                    f.write(f"Original Timestamp: {scene_graph_data.get('timestamp', 'N/A')}\n")
                    f.write("\n")
                    
                    # Main object
                    if 'main_object' in scene_graph_data:
                        main_obj = scene_graph_data['main_object']
                        f.write("MAIN OBJECT:\n")
                        f.write(f"  Name: {main_obj.get('name', 'N/A')}\n")
                        f.write(f"  ID: {main_obj.get('id', 'N/A')}\n")
                        
                        if 'attributes' in main_obj:
                            attrs = main_obj['attributes']
                            f.write("  Attributes:\n")
                            f.write(f"    Color: {attrs.get('color', 'N/A')}\n")
                            f.write(f"    Material: {attrs.get('material', 'N/A')}\n")
                            f.write(f"    Style: {attrs.get('style', 'N/A')}\n")
                        f.write("\n")
                    
                    # Parts
                    if 'parts' in scene_graph_data and scene_graph_data['parts']:
                        f.write(f"PARTS ({len(scene_graph_data['parts'])}):\n")
                        for j, part in enumerate(scene_graph_data['parts']):
                            f.write(f"  Part {j+1}:\n")
                            f.write(f"    Name: {part.get('name', 'N/A')}\n")
                            f.write(f"    Relationship: {part.get('relationship_to_main', 'N/A')}\n")
                            if 'attributes' in part:
                                attrs = part['attributes']
                                f.write(f"    Color: {attrs.get('color', 'N/A')}\n")
                                f.write(f"    Material: {attrs.get('material', 'N/A')}\n")
                                f.write(f"    Style: {attrs.get('style', 'N/A')}\n")
                            f.write("\n")
                    
                    # Environment
                    if 'environment' in scene_graph_data and scene_graph_data['environment']:
                        f.write(f"ENVIRONMENT ({len(scene_graph_data['environment'])}):\n")
                        for j, env in enumerate(scene_graph_data['environment']):
                            f.write(f"  Environment {j+1}:\n")
                            f.write(f"    Name: {env.get('name', 'N/A')}\n")
                            f.write(f"    ID: {env.get('id', 'N/A')}\n")
                            if 'attributes' in env:
                                attrs = env['attributes']
                                f.write(f"    Color: {attrs.get('color', 'N/A')}\n")
                                f.write(f"    Material: {attrs.get('material', 'N/A')}\n")
                                f.write(f"    Style: {attrs.get('style', 'N/A')}\n")
                            f.write("\n")
                    
                    # Spatial context
                    if 'spatial_context' in scene_graph_data:
                        spatial = scene_graph_data['spatial_context']
                        f.write("SPATIAL CONTEXT:\n")
                        f.write(f"  Position: {spatial.get('position', 'N/A')}\n")
                        nearby_objs = spatial.get('nearby_objects', [])
                        if nearby_objs:
                            f.write(f"  Nearby Objects ({len(nearby_objs)}): {', '.join(nearby_objs)}\n")
                        else:
                            f.write("  Nearby Objects: None\n")
                        f.write("\n")
                    
                    # JSON representation
                    f.write("RAW JSON DATA:\n")
                    f.write(json.dumps(scene_graph_data, indent=2, ensure_ascii=False))
                    f.write("\n\n")
                    f.write("=" * 80 + "\n\n")
                
                # Summary
                f.write("SUMMARY:\n")
                f.write(f"Total Objects: {len(self.loaded_scene_graphs)}\n")
                
                # Count statistics
                main_objects = [sg for sg in self.loaded_scene_graphs if 'main_object' in sg and sg['main_object'].get('name')]
                parts_count = sum(len(sg.get('parts', [])) for sg in self.loaded_scene_graphs)
                env_count = sum(len(sg.get('environment', [])) for sg in self.loaded_scene_graphs)
                spatial_count = len([sg for sg in self.loaded_scene_graphs if 'spatial_context' in sg and sg['spatial_context'].get('position')])
                
                f.write(f"Objects with Main Object: {len(main_objects)}\n")
                f.write(f"Total Parts: {parts_count}\n")
                f.write(f"Total Environment Objects: {env_count}\n")
                f.write(f"Objects with Spatial Context: {spatial_count}\n")
                f.write("\n")
            
            rospy.loginfo(f"Successfully saved loaded scene graphs to {filename}")
            
        except Exception as e:
            rospy.logerr(f"Error saving loaded scene graphs as txt: {e}")
            import traceback
            traceback.print_exc()

    def replay_saved_scene_graphs(self, trigger_msg: ObjectDescriptions):
        """Replay saved ObjectSceneGraph data instead of processing"""
        if not self.loaded_scene_graphs:
            rospy.logwarn("No saved ObjectSceneGraph data to replay")
            return
        
        # Map incoming object IDs to saved data (for now, just cycle through saved data)
        for i, description in enumerate(trigger_msg.objects):
            if i < len(self.loaded_scene_graphs):
                saved_data = self.loaded_scene_graphs[i]
                
                # Convert saved data back to ROS message
                scene_graph_msg = self.dict_to_scene_graph_msg(saved_data, description.object_id)
                
                # Publish the replayed message
                self.result_pub.publish(scene_graph_msg)
                rospy.loginfo(f"Replayed ObjectSceneGraph for object {description.object_id} from saved data")
            else:
                rospy.logwarn(f"No saved data available for object {description.object_id} (index {i})")

    def dict_to_scene_graph_msg(self, scene_graph_data, object_id):
        """Convert dictionary back to ObjectSceneGraph ROS message"""
        msg = ObjectSceneGraph()
        
        # Main object
        if 'main_object' in scene_graph_data:
            main_obj = scene_graph_data['main_object']
            msg.main_object.id = object_id  # Use current object_id
            msg.main_object.name = main_obj.get('name', '')
            
            # Attributes
            if 'attributes' in main_obj:
                attrs = main_obj['attributes']
                msg.main_object.attributes.color = attrs.get('color', '')
                msg.main_object.attributes.material = attrs.get('material', '')
                msg.main_object.attributes.style = attrs.get('style', '')
        
        # Parts
        if 'parts' in scene_graph_data:
            for part_data in scene_graph_data['parts']:
                part = ObjectPart()
                part.name = part_data.get('name', '')
                part.relationship_to_main = part_data.get('relationship_to_main', '')
                
                if 'attributes' in part_data:
                    attrs = part_data['attributes']
                    part.attributes.color = attrs.get('color', '')
                    part.attributes.material = attrs.get('material', '')
                    part.attributes.style = attrs.get('style', '')
                
                msg.parts.append(part)
        
        # Environment
        if 'environment' in scene_graph_data:
            for env_data in scene_graph_data['environment']:
                env_obj = ObjectInfo()
                env_obj.id = env_data.get('id', 0)
                env_obj.name = env_data.get('name', '')
                
                if 'attributes' in env_data:
                    attrs = env_data['attributes']
                    env_obj.attributes.color = attrs.get('color', '')
                    env_obj.attributes.material = attrs.get('material', '')
                    env_obj.attributes.style = attrs.get('style', '')
                
                msg.environment.append(env_obj)
        
        # Spatial context
        if 'spatial_context' in scene_graph_data:
            spatial = scene_graph_data['spatial_context']
            msg.spatial_context.position = spatial.get('position', '')
            msg.spatial_context.nearby_objects = spatial.get('nearby_objects', [])
        
        return msg

    def batch_processing_loop(self):
        """Main batch processing loop running in separate thread"""
        # Skip batch processing in load mode
        if self.load_mode:
            rospy.loginfo("Batch processing disabled in load mode")
            return
        
        while self.running and not rospy.is_shutdown():
            try:
                # Check if we should process a batch
                should_process = False
                current_time = time.time()
                
                with self.queue_lock:
                    queue_size = len(self.request_queue)
                    
                    # Process if we have enough requests or timeout elapsed
                    if queue_size >= self.batch_size:
                        should_process = True
                        rospy.loginfo(f"Processing batch: queue full ({queue_size} items)")
                    elif (queue_size > 0 and self.last_request_time and 
                          current_time - self.last_request_time >= self.batch_timeout):
                        should_process = True
                        rospy.loginfo(f"Processing batch: timeout reached ({queue_size} items)")
                
                if should_process:
                    self.process_batch()
                else:
                    # Sleep briefly to avoid busy waiting
                    time.sleep(0.1)
                    
            except Exception as e:
                rospy.logerr(f"Error in batch processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)  # Prevent rapid error loops

    def process_batch(self):
        """Process a batch of requests"""
        batch_requests = []
        
        # Extract batch from queue
        with self.queue_lock:
            batch_size = min(self.batch_size, len(self.request_queue))
            for _ in range(batch_size):
                if self.request_queue:
                    batch_requests.append(self.request_queue.popleft())
        
        if not batch_requests:
            return
        
        rospy.loginfo(f"Processing batch of {len(batch_requests)} requests")
        
        try:
            # Prepare prompts for batch processing
            user_prompts = [req['caption'] for req in batch_requests]
            
            # Generate batch responses
            responses = self.generate_batch_responses(user_prompts)
            
            # Process and publish results
            for i, (request, response) in enumerate(zip(batch_requests, responses)):
                try:
                    # Parse JSON response
                    if isinstance(response, str):
                        json_content = self.parse_json_response(response)
                    else:
                        json_content = response
                    
                    # Convert to ROS message and publish
                    scene_graph_msg = self.json_to_scene_graph_msg(json_content, request['object_id'])
                    self.result_pub.publish(scene_graph_msg)
                    
                    # Save to disk if in dump mode
                    if self.dump_mode:
                        self.save_scene_graph(scene_graph_msg, request['object_id'])
                    
                    rospy.loginfo(f"Published result for object {request['object_id']}")
                    
                except Exception as e:
                    rospy.logerr(f"Error processing result for object {request['object_id']}: {e}")
                    
        except Exception as e:
            rospy.logerr(f"Error in batch processing: {e}")
            import traceback
            traceback.print_exc()

    def parse_json_response(self, response_text):
        """Parse JSON response from LLM output"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            rospy.logwarn(f"JSON Parse Error: {e}")
            # Try to extract JSON from response if it's embedded
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"error": "Failed to parse JSON", "content": response_text}
            else:
                return {"error": "No JSON found", "content": response_text}

    def json_to_scene_graph_msg(self, json_data, object_id):
        """Convert JSON response to ObjectSceneGraph ROS message"""
        msg = ObjectSceneGraph()
        
        # Main object
        if 'main_object' in json_data:
            main_obj = json_data['main_object']
            msg.main_object.id = object_id
            msg.main_object.name = main_obj.get('name', '')
            
            # Attributes
            if 'attributes' in main_obj:
                attrs = main_obj['attributes']
                msg.main_object.attributes.color = attrs.get('color', '')
                msg.main_object.attributes.material = attrs.get('material', '')
                msg.main_object.attributes.style = attrs.get('style', '')
        
        # Parts
        if 'parts' in json_data:
            for part_data in json_data['parts']:
                part = ObjectPart()
                part.name = part_data.get('name', '')
                part.relationship_to_main = part_data.get('relationship_to_main', '')
                
                if 'attributes' in part_data:
                    attrs = part_data['attributes']
                    part.attributes.color = attrs.get('color', '')
                    part.attributes.material = attrs.get('material', '')
                    part.attributes.style = attrs.get('style', '')
                
                msg.parts.append(part)
        
        # Environment
        if 'environment' in json_data:
            for env_data in json_data['environment']:
                env_obj = ObjectInfo()
                env_obj.id = 0  # Environment objects don't have specific IDs
                env_obj.name = env_data.get('name', '')
                
                if 'attributes' in env_data:
                    attrs = env_data['attributes']
                    env_obj.attributes.color = attrs.get('color', '')
                    env_obj.attributes.material = attrs.get('material', '')
                    env_obj.attributes.style = attrs.get('style', '')
                
                msg.environment.append(env_obj)
        
        # Spatial context
        if 'spatial_context' in json_data:
            spatial = json_data['spatial_context']
            msg.spatial_context.position = spatial.get('position', '')
            msg.spatial_context.nearby_objects = spatial.get('nearby_objects', [])
        
        return msg

    def shutdown(self, signum, frame):
        """Gracefully shut down the node."""
        rospy.loginfo("Shutting down LLM Parser Node gracefully...")
        self.running = False  # Stop the main loop
        
        # Wait for processing thread to finish
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            rospy.loginfo("Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=5.0)
        
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                # vLLM doesn't have a shutdown method, but we can delete the object
                del self.llm
                rospy.loginfo("vLLM engine shut down successfully.")
            except Exception as e:
                rospy.logerr(f"Error shutting down vLLM engine: {e}")
        
        rospy.loginfo("LLM Parser Node shutdown complete.")
        sys.exit(0)

    
    
    def generate_response_second(self, user_prompt):
        """Generate JSON response using vLLM with optimized parameters"""
        if self.load_mode or self.llm is None:
            rospy.logwarn("Cannot generate response in load mode or without LLM")
            return {"error": "LLM not available"}
        
        start_time = time.time()
        
        # Format prompt using tokenizer's chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode if supported
            )
        except:
            # Fallback if enable_thinking parameter doesn't exist
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Generate response using vLLM
        outputs = self.llm.generate([prompt], self.sampling_params_second)
        content = outputs[0].outputs[0].text.strip()
        
        # Parse JSON response
        print("Content received from vLLM:", content)
        try:
            json_content = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            # Try to extract JSON from response if it's embedded
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    json_content = json.loads(json_match.group())
                except json.JSONDecodeError:
                    json_content = {"error": "Failed to parse JSON", "content": content}
            else:
                json_content = {"error": "No JSON found", "content": content}
        
        total_time = time.time() - start_time
        print(f"[vLLM Timing] Total: {total_time:.3f}s")
        
        return json_content
    
    def generate_batch_responses(self, user_prompts):
        """Generate responses for a batch of user prompts."""
        if self.load_mode or self.llm is None:
            rospy.logwarn("Cannot generate batch responses in load mode or without LLM")
            return [{"error": "LLM not available"} for _ in user_prompts]
        
        start_time = time.time()

        # Format prompts
        prompts = []
        for user_prompt in user_prompts:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except:
                # Fallback if enable_thinking parameter doesn't exist
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            prompts.append(prompt)

        # Generate responses in batch
        outputs = self.llm.generate(prompts, self.sampling_params_second)

        # Extract and return responses
        responses = [output.outputs[0].text.strip() for output in outputs]
        total_time = time.time() - start_time
        rospy.loginfo(f"[vLLM Batch Timing] Total: {total_time:.3f}s for {len(user_prompts)} requests")
        return responses

    def run(self):
        """Main execution loop for ROS node"""
        # Register the shutdown handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.shutdown)
        
        rospy.loginfo("LLM Parser Node is running...")
        rospy.loginfo("Waiting for ObjectDescriptions messages on /scene_graph/parser_llm")
        rospy.loginfo("Publishing results to /scene_graph/parser_llm/result")
        
        try:
            # Keep the node running
            rospy.spin()
        except KeyboardInterrupt:
            self.shutdown(None, None)
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            self.shutdown(None, None)

if __name__ == "__main__":
    try:
        node = QwenChatNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("LLM Parser Node interrupted")
    except Exception as e:
        rospy.logerr(f"Failed to start LLM Parser Node: {e}")
        import traceback
        traceback.print_exc()
