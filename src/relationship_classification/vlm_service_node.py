#!/usr/bin/env python3

import rospy
import cv2
import base64
import json
import requests
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from scene_graph.srv import VLMInference, VLMInferenceResponse
from scene_graph.msg import ObjectSceneGraph, ObjectInfo, ObjectAttribute, ObjectPart, ObjectSpatialContext
from std_msgs.msg import String
import sys
import os
import time


class VLMServiceNode:
    def __init__(self):
        rospy.init_node('vlm_service_node', anonymous=True)
        
        # Get parameters
        self.backend = rospy.get_param('~backend', 'florence')  # florence, vLLM, external
        self.api_key = rospy.get_param('~api_key', '')  # For external services (not needed for containerized Google API)
        self.vlm_url = rospy.get_param('~vlm_url', 'http://localhost:8002/v1/chat/completions')  # vLLM server URL   /chat/completions
        self.florence_url = rospy.get_param('~florence_url', 'http://florence:8001')  # Florence server URL
        self.external_url = rospy.get_param('~external_url', 'http://localhost:8002/v1')  # Google API container URL
        self.external_model = rospy.get_param('~external_model', 'gemini-2.5-flash-lite')  # External model name
        self.external_rate_limit = rospy.get_param('~external_rate_limit', 4.0)  # Rate limit for external API in seconds
        
        # Variables
        self.index = 0  # For assigning unique IDs to objects
        
        # Persistent object tracking
        self.known_objects = []  # List of known objects with their features
        self.next_persistent_id = 1  # Counter for persistent IDs
        
        # Object similarity thresholds
        self.position_threshold = 50.0  # Pixel distance threshold for position similarity
        self.name_match_weight = 0.7   # Weight for name matching
        self.position_match_weight = 0.3  # Weight for position matching
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize external API client (containerized Google API)
        self.external_client_ready = False
        self.last_external_request_time = 0.0  # Track last external API request time
        self.external_request_interval = self.external_rate_limit  # Use configurable rate limit
        if self.backend == 'external':
            # No API key needed - using containerized Google API
            self.external_client_ready = True
            rospy.loginfo(f"External API ready to use containerized Google API at: {self.external_url}")
        
        # Create service
        self.service = rospy.Service('vlm_inference', VLMInference, self.handle_vlm_inference)
        
        # Create publisher for object relationships
        self.relationships_pub = rospy.Publisher('/scene_graph/object_relationships', String, queue_size=10)
        
        # Define the detection prompt
        #- Unique object detection is required (dont repeat objects with identical bounding boxes)
        #- Maximum 10 unique objects can be detected
        self.detection_prompt = """Analyze this image and detect all objects. You must respond with ONLY valid JSON in the exact format below, with no additional text, explanations, or markdown formatting.

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
- Bounding box coordinates are percentile values from 0 to 999 (0 = top/left, 999 = bottom/right)
- Use complete attribute values, never "..." or truncation
- Include empty string "" for unknown attributes
- Relations describe spatial relationships like: "on", "under", "attached_to", "in", "at"
- Response must be parseable JSON only

Detect all objects in the image and respond with the JSON:"""
        
        rospy.loginfo(f"VLM Service Node started with backend: {self.backend}")
        
    def image_to_base64(self, image_msg):
        """Convert ROS Image message to base64 encoded string"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
            
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', cv_image)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64
            
        except Exception as e:
            rospy.logerr(f"Error converting image to base64: {e}")
            return None
    
    def call_florence_api(self, image_base64):
        """Call Florence container API"""
        try:
            # Prepare request for Florence API (adjust based on actual Florence API)
            payload = {
                "image": image_base64,
                "task": "object_detection"
            }
            
            response = requests.post(
                f"{self.florence_url}/detect",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                rospy.logerr(f"Florence API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error calling Florence API: {e}")
            return None
    
    def call_vllm_api(self, image_base64):
        """Call vLLM API using vLLM-specific format for vision models"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # vLLM-specific format for vision models - use string content with image data
            payload = {
                "model": "OpenGVLab/InternVL3_5-4B-HF",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.detection_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.vlm_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=240  # Reduced timeout for efficiency
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return self.parse_json_response(content)
            else:
                rospy.logerr(f"vLLM API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error calling vLLM API: {e}")
            return None
    
    def call_external_api(self, image_base64):
        """Call external containerized Google API using OpenAI-compatible format"""
        try:
            if not self.external_client_ready:
                rospy.logerr("External API client not ready")
                return None
            
            # Rate limiting: ensure configured seconds have passed since last request   TODO:change for other implementations
            import time
            current_time = time.time()
            time_since_last_request = current_time - self.last_external_request_time
            
            if time_since_last_request < self.external_request_interval:
                wait_time = self.external_request_interval - time_since_last_request
                rospy.loginfo(f"Rate limiting: waiting {wait_time:.2f} seconds before next external API call")
                time.sleep(wait_time)
            
            # Update the last request time
            self.last_external_request_time = time.time()
            
            # Prepare OpenAI-compatible request for containerized Google API
            headers = {
                "Content-Type": "application/json"
            }
            
            # OpenAI-compatible format
            payload = {
                "model": self.external_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.detection_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7
            }

            
            rospy.loginfo("Sending request to containerized Google API...")
            
            # Track request time
            request_start_time = time.time()
            
            response = requests.post(
                f"{self.external_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=360  # Reasonable timeout for Google API
            )
            
            # Calculate request duration
            request_duration = time.time() - request_start_time
            rospy.loginfo(f"External API request completed in {request_duration:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Log performance info if available
                if "performance" in result:
                    perf = result["performance"]
                    rospy.loginfo(f"External API inference time: {perf.get('inference_time_seconds', 'N/A')}s")
                    rospy.loginfo(content)
                
                return self.parse_json_response(content)
            elif response.status_code == 429:
                # Handle rate limiting from the container
                rospy.logwarn("Rate limit exceeded on external API container")
                return None
            else:
                rospy.logerr(f"External API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error calling external containerized API: {e}")
            return None
    
    def parse_json_response(self, response_text):
        """Parse JSON response from VLM"""
        try:
            # Clean up response text (remove markdown code blocks if present)
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            return json.loads(cleaned_text)
            
        except json.JSONDecodeError as e:
            rospy.logerr(f"JSON parsing error: {e}")
            rospy.logerr(f"Response text: {response_text[:500]}...")
            return None
    
    def calculate_object_similarity(self, obj1, obj2):
        """Calculate similarity between two objects based on name and position"""
        # Name similarity (exact match or contains)
        name1 = obj1.get("label", "").lower()
        name2 = obj2.get("label", "").lower()
        
        name_score = 0.0
        if name1 == name2:
            name_score = 1.0
        elif name1 in name2 or name2 in name1:
            name_score = 0.8
        elif any(word in name2.split() for word in name1.split()) or any(word in name1.split() for word in name2.split()):
            name_score = 0.6
        
        # Position similarity (center of bounding boxes)
        bbox1 = obj1.get("bounding_box", [0, 0, 0, 0])
        bbox2 = obj2.get("bounding_box", [0, 0, 0, 0])
        
        center1_x = (bbox1[0] + bbox1[2]) / 2.0
        center1_y = (bbox1[1] + bbox1[3]) / 2.0
        center2_x = (bbox2[0] + bbox2[2]) / 2.0
        center2_y = (bbox2[1] + bbox2[3]) / 2.0
        
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        position_score = max(0.0, 1.0 - (distance / self.position_threshold))
        
        # Combined similarity score
        total_score = (name_score * self.name_match_weight + 
                      position_score * self.position_match_weight)
        
        return total_score
    
    def find_or_create_persistent_id(self, obj_data):
        """Find existing object or create new persistent ID"""
        best_match_id = None
        best_similarity = 0.0
        similarity_threshold = 0.6  # Minimum similarity to consider a match
        
        # Check against known objects
        for known_obj in self.known_objects:
            similarity = self.calculate_object_similarity(obj_data, known_obj["data"])
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match_id = known_obj["persistent_id"]
        
        if best_match_id is not None:
            # Update the known object with current data
            for known_obj in self.known_objects:
                if known_obj["persistent_id"] == best_match_id:
                    known_obj["data"] = obj_data.copy()
                    known_obj["last_seen"] = time.time()
                    break
            return best_match_id
        else:
            # Create new persistent ID
            new_id = self.next_persistent_id
            self.next_persistent_id += 1
            
            # Add to known objects
            self.known_objects.append({
                "persistent_id": new_id,
                "data": obj_data.copy(),
                "last_seen": time.time()
            })
            
            return new_id
    
    def cleanup_old_objects(self, max_age_seconds=300):
        """Remove objects that haven't been seen for a while"""
        current_time = time.time()
        self.known_objects = [obj for obj in self.known_objects 
                            if current_time - obj["last_seen"] < max_age_seconds]
    
    def create_object_scene_graph(self, detection_result):
        """Convert detection result to ObjectSceneGraph message"""
        try:
            if not detection_result or "objects" not in detection_result:
                return []
            
            # Clean up old objects periodically TODO: why?
            self.cleanup_old_objects()
            
            scene_graphs = []
            object_mapping = {}  # Map from object name to persistent ID
            relations_data = []  # Store relations data separately
            
            # First pass: Create all objects with persistent IDs
            for obj_data in detection_result["objects"]:
                scene_graph = ObjectSceneGraph()
                
                # Get or create persistent ID
                persistent_id = self.find_or_create_persistent_id(obj_data)
                
                # Create main object
                scene_graph.main_object = ObjectInfo()
                scene_graph.main_object.id = persistent_id
                scene_graph.main_object.name = obj_data.get("label", "unknown")
                
                # Store mapping from object name to persistent ID
                object_mapping[scene_graph.main_object.name] = persistent_id
                
                # Store relations data for later mapping
                relations_data.append({
                    "object_id": persistent_id,
                    "object_name": scene_graph.main_object.name,
                    "relations": obj_data.get("relations", [])
                })
                
                # Set attributes
                scene_graph.main_object.attributes = ObjectAttribute()
                attributes = obj_data.get("attributes", {})
                scene_graph.main_object.attributes.color = attributes.get("color", "")
                scene_graph.main_object.attributes.style = attributes.get("style", "")
                
                # Create spatial context (simplified)
                scene_graph.spatial_context = ObjectSpatialContext()
                bbox = obj_data.get("bounding_box", [0, 0, 0, 0])
                
                # Set position as string (you can format this as needed)
                scene_graph.spatial_context.position = f"bbox_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                
                # Set 2D bounding box in image coordinates
                scene_graph.bbox_2d = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                
                # Set nearby objects from relations (names for now, will be mapped later)
                scene_graph.spatial_context.nearby_objects = []
                relations = obj_data.get("relations", [])
                for relation in relations:
                    target = relation.get("target", "")
                    if target:
                        scene_graph.spatial_context.nearby_objects.append(target)
                
                # Add relations as environment objects
                scene_graph.environment = []
                relations = obj_data.get("relations", [])
                for relation in relations:
                    env_obj = ObjectInfo()
                    env_obj.id = -1  # Mark as relation target
                    env_obj.name = relation.get("target", "")
                    env_obj.attributes = ObjectAttribute()
                    env_obj.attributes.color = ""
                    env_obj.attributes.style = relation.get("type", "")
                    scene_graph.environment.append(env_obj)
                
                # Initialize empty parts list
                scene_graph.parts = []
                
                scene_graphs.append(scene_graph)
            
            # Second pass: Map relationships and publish them
            self.map_and_publish_relationships(relations_data, object_mapping)
                
            return scene_graphs
            
        except Exception as e:
            rospy.logerr(f"Error creating ObjectSceneGraph: {e}")
            return []
    
    def map_and_publish_relationships(self, relations_data, object_mapping):
        """Map relation targets to actual object IDs and publish relationships"""
        try:
            relationships = {}
            
            for obj_relations in relations_data:
                source_id = obj_relations["object_id"]
                source_name = obj_relations["object_name"]
                relations = obj_relations["relations"]
                
                for relation in relations:
                    target_name = relation.get("target", "")
                    relation_type = relation.get("type", "")
                    
                    if target_name and relation_type:
                        # Try to find target object ID by exact name match
                        target_id = object_mapping.get(target_name)
                        
                        # If exact match not found, try partial matching
                        if target_id is None:
                            for obj_name, obj_id in object_mapping.items():
                                if target_name.lower() in obj_name.lower() or obj_name.lower() in target_name.lower():
                                    target_id = obj_id
                                    break
                        
                        if target_id is not None:
                            # Add relationship - store just the relationship type as string
                            if source_id not in relationships:
                                relationships[source_id] = {}
                            relationships[source_id][target_id] = relation_type
                        else:
                            rospy.logwarn(f"Could not find target object '{target_name}' for relationship from '{source_name}'")
            
            # Publish relationships
            if relationships:
                relationships_msg = String()
                relationships_msg.data = json.dumps(relationships)
                self.relationships_pub.publish(relationships_msg)
                rospy.loginfo(f"Published {len(relationships)} object relationships")
                    
        except Exception as e:
            rospy.logerr(f"Error mapping and publishing relationships: {e}")
    
    def handle_vlm_inference(self, req):
        """Handle VLM inference service request"""
        response = VLMInferenceResponse()
        
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(req.image)
            if not image_base64:
                response.success = False
                response.error_message = "Failed to convert image to base64"
                return response
            
            # Call appropriate backend
            if self.backend == "florence":
                detection_result = self.call_florence_api(image_base64)
            elif self.backend == "vLLM":
                detection_result = self.call_vllm_api(image_base64)
            elif self.backend == "external":
                detection_result = self.call_external_api(image_base64)
            else:
                response.success = False
                response.error_message = f"Unknown backend: {self.backend}"
                return response
            
            if detection_result is None:
                response.success = False
                response.error_message = f"API call to {self.backend} failed"
                return response
            
            # Convert to ObjectSceneGraph messages
            scene_graphs = self.create_object_scene_graph(detection_result)
            
            response.scene_graphs = scene_graphs
            response.success = True
            response.error_message = ""
            
            rospy.loginfo(f"Successfully processed image with {len(scene_graphs)} objects detected")
            
        except Exception as e:
            rospy.logerr(f"Error in VLM inference: {e}")
            response.success = False
            response.error_message = str(e)
        
        return response


def main():
    try:
        # Check for required parameters based on backend
        backend = rospy.get_param('~backend', 'florence')
        
        if backend not in ['florence', 'vLLM', 'external']:
            rospy.logerr(f"Invalid backend parameter: {backend}. Must be 'florence', 'vLLM', or 'external'")
            sys.exit(1)
        
        if backend == 'external':
            # No API key needed for containerized Google API
            external_url = rospy.get_param('~external_url', 'http://localhost:8002/v1')
            rospy.loginfo(f"Using containerized Google API at: {external_url}")
            
            # Test connection to containerized API
            try:
                import requests
                test_response = requests.get(f"{external_url.rstrip('/v1')}/health", timeout=5)
                if test_response.status_code != 200:
                    rospy.logwarn(f"External API health check failed: {test_response.status_code}")
            except Exception as e:
                rospy.logwarn(f"Could not reach external API container: {e}")
        
        # Create and run the service node
        service_node = VLMServiceNode()
        rospy.loginfo("VLM Service Node ready")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("VLM Service Node shutting down")
    except Exception as e:
        rospy.logerr(f"Error starting VLM Service Node: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()