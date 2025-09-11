#!/usr/bin/env python3

import rospy
import numpy as np
import traceback
from scipy.spatial import KDTree
import time
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading
import signal
import sys

# ROS imports
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point, Point32
from scene_graph.msg import DetectedObjects, DetectedObject, GraphObjects, GraphObject, ObjectDescription, ObjectDescriptions, ObjectSceneGraph, ObjectAttribute, ObjectInfo, ObjectPart, ObjectSpatialContext
from scene_graph.srv import RegisterObject, RegisterObjectResponse, QueryObjects, QueryObjectsResponse, GetObjectInfo, GetObjectInfoResponse

@dataclass
class ObjectInstance:
    """Represents a persistent object instance in the spatial tracker"""
    object_id: int
    label: str
    world_position: np.ndarray
    confidence: float
    first_detection_time: float
    last_detection_time: float
    detection_count: int = 1
    object_size: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.2]))  # Default 20cm cube
    
    # New attributes for LLM parser results
    llm_parsed: bool = False
    llm_request_sent: bool = False
    raw_description: str = ""
    parsed_json: dict = field(default_factory=dict)
    main_object: dict = field(default_factory=dict)
    parts: List[dict] = field(default_factory=list)
    environment: List[dict] = field(default_factory=list)
    spatial_context: dict = field(default_factory=dict)
    
    # New attributes for object relationship mapping
    nearby_objects: Dict[int, str] = field(default_factory=dict)  # object_id -> object_name
    mapped_relationships: Dict[int, str] = field(default_factory=dict)  # object_id -> relationship_description
    
    @property
    def name(self) -> str:
        """Get object name - alias for label for backward compatibility"""
        return self.label

class DetectedObjectRegistry:
    """Registry for managing detected objects with spatial indexing"""
    
    def __init__(self, position_threshold: float = 1.5, confidence_threshold: float = 0.3):
        self.objects: Dict[int, ObjectInstance] = {}
        self.next_id = 1
        self.position_threshold = position_threshold
        self.confidence_threshold = confidence_threshold
        
        # Spatial indexing
        self.kdtree = None
        self.position_array = []
        self.id_array = []
        
        # Thread safety
        self.lock = threading.Lock()
    
    def cleanup(self):
        """Clean up resources"""
        with self.lock:
            self.objects.clear()
            self.kdtree = None
            self.position_array = []
            self.id_array = []
    
    def rebuild_spatial_index(self):
        """Rebuild KD-tree spatial index"""
        with self.lock:
            if len(self.objects) == 0:
                self.kdtree = None
                self.position_array = np.array([]).reshape(0, 3)
                self.id_array = np.array([], dtype=int)
                return
            
            positions = []
            ids = []
            
            for obj_id, obj_instance in self.objects.items():
                positions.append(obj_instance.world_position)
                ids.append(obj_id)
            
            self.position_array = np.array(positions)
            self.id_array = np.array(ids)
            
            if len(positions) > 0:
                self.kdtree = KDTree(self.position_array)
    
    def query_objects_spatial(self, center: np.ndarray, radius: float) -> List[int]:
        """Query objects within radius of center point"""
        if self.kdtree is None:
            return []
        
        try:
            indices = self.kdtree.query_ball_point(center, radius)
            # Convert numpy.int64 to Python int to avoid JSON serialization issues
            return [int(self.id_array[i]) for i in indices]
        except Exception as e:
            rospy.logwarn(f"Error in spatial query: {e}")
            return []
    
    def query_objects_by_label(self, label: str) -> List[int]:
        """Query objects by semantic label"""
        result = []
        with self.lock:
            for obj_id, obj_instance in self.objects.items():
                if obj_instance.label.lower() == label.lower():
                    result.append(obj_id)
        return result
    
    def get_object_count(self) -> int:
        """Get total number of tracked objects"""
        return len(self.objects)
    
    def register_or_update_object(self, label: str, world_position: np.ndarray, 
                                confidence: float, bbox: Tuple[float, float, float, float],
                                camera_pose: np.ndarray, object_size: np.ndarray = None) -> int:
        """Register new object or update existing one"""
        
        # Find nearest objects with same label
        min_distance = float('inf')
        closest_obj_id = None
        
        with self.lock:
            for obj_id, obj_instance in self.objects.items():
                if obj_instance.label == label:
                    distance = np.linalg.norm(world_position - obj_instance.world_position)
                    if distance < min_distance:
                        min_distance = distance
                        closest_obj_id = obj_id
        
        if closest_obj_id is not None and min_distance <= self.position_threshold:
            # Update existing object
            with self.lock:
                obj_instance = self.objects[closest_obj_id]
                obj_instance.detection_count += 1
                obj_instance.last_detection_time = time.time()
                
                # Update position and confidence with weighted average
                if confidence > obj_instance.confidence:
                    alpha = 0.7  # Weight for new detection
                    obj_instance.world_position = (alpha * world_position + 
                                                  (1 - alpha) * obj_instance.world_position)
                    obj_instance.confidence = confidence
                    
                    # Update object size if provided
                    if object_size is not None:
                        obj_instance.object_size = self.sanitize_object_size(object_size)
            
            self.rebuild_spatial_index()
            rospy.logdebug(f"Updated {label} (ID: {closest_obj_id}) at distance {min_distance:.2f}m")
            return closest_obj_id
        
        else:
            # Create new object
            current_time = time.time()
            
            with self.lock:
                obj_id = self.next_id
                self.next_id += 1
                
                obj_instance = ObjectInstance(
                    object_id=obj_id,
                    label=label,
                    world_position=world_position.copy(),
                    confidence=confidence,
                    first_detection_time=current_time,
                    last_detection_time=current_time,
                    object_size=self.sanitize_object_size(object_size.copy() if object_size is not None else np.array([0.2, 0.2, 0.2]))
                )
                
                self.objects[obj_id] = obj_instance
            
            self.rebuild_spatial_index()
            rospy.loginfo(f"Registered new {label} (ID: {obj_id}) at ({world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f})")
            return obj_id
    
    def create_3d_bounding_box(self, position: np.ndarray, size: np.ndarray) -> List[Point32]:
        """Create 3D bounding box with min and max corners for graph management node"""
        # Sanitize size to ensure valid dimensions
        size = self.sanitize_object_size(size)
        
        # Calculate half extents
        half_x, half_y, half_z = size / 2.0
        
        # Create only min and max corners (2 points) as expected by graph management node
        min_corner = Point32(
            x=position[0] - half_x, 
            y=position[1] - half_y, 
            z=position[2] - half_z
        )
        max_corner = Point32(
            x=position[0] + half_x, 
            y=position[1] + half_y, 
            z=position[2] + half_z
        )
        
        return [min_corner, max_corner]

    def sanitize_object_size(self, size: np.ndarray) -> np.ndarray:
        """Ensure object size has valid positive dimensions"""
        if size is None or len(size) != 3:
            return np.array([0.2, 0.2, 0.2])  # Default 20cm cube
        
        # Ensure minimum dimensions to prevent zero-size bounding boxes
        min_dimension = 0.05  # 5cm minimum
        size = np.maximum(size, min_dimension)
        
        # Ensure no NaN or inf values
        size = np.nan_to_num(size, nan=0.2, posinf=1.0, neginf=0.05)
        
        return size
    
    def find_objects_by_label_prefix(self, label_prefix: str) -> List[Tuple[int, 'ObjectInstance']]:
        """Find all objects whose labels start with the given prefix"""
        matching_objects = []
        label_prefix_lower = label_prefix.lower()
        
        for obj_id, obj_instance in self.objects.items():
            if obj_instance.label.lower().startswith(label_prefix_lower):
                matching_objects.append((obj_id, obj_instance))
        
        return matching_objects
    
    def get_object_by_id(self, object_id: int) -> Optional['ObjectInstance']:
        """Get object instance by ID"""
        return self.objects.get(object_id)
    
    def get_all_objects(self) -> Dict[int, 'ObjectInstance']:
        """Get all tracked objects"""
        return self.objects.copy()
    
    def get_object_count(self) -> int:
        """Get the number of tracked objects"""
        return len(self.objects)
    
    def find_nearby_objects(self, center_object_id: int, search_radius: float = 3.0) -> Dict[int, str]:
        """Find objects within search radius of a given object and return their ID->name mapping"""
        nearby_objects = {}
        
        # Get the center object
        center_object = self.objects.get(center_object_id)
        if not center_object:
            rospy.logwarn(f"Center object {center_object_id} not found")
            return nearby_objects
        
        center_position = center_object.world_position
        
        # Query spatial index for nearby objects
        nearby_ids = self.query_objects_spatial(center_position, search_radius)
        
        # Create ID -> name mapping, excluding the center object itself
        for obj_id in nearby_ids:
            if obj_id != center_object_id:
                obj_instance = self.objects.get(obj_id)
                if obj_instance:
                    nearby_objects[obj_id] = obj_instance.label
                    
        rospy.logdebug(f"Found {len(nearby_objects)} nearby objects for object {center_object_id} within {search_radius}m")
        return nearby_objects
    
    def map_name_to_objects(self, target_name: str, nearby_objects: Dict[int, str]) -> List[int]:
        """Map a name string to object IDs from nearby objects dictionary"""
        matched_ids = []
        target_name_lower = target_name.strip().lower()
        
        if not target_name_lower:
            return matched_ids
        
        for obj_id, obj_name in nearby_objects.items():
            obj_name_lower = obj_name.lower()
            
            # Split target name into words
            target_words = target_name_lower.split()
            
            if len(target_words) == 1:
                # Single word: check if it's part of the object name
                if target_words[0] in obj_name_lower:
                    matched_ids.append(obj_id)
                    rospy.logdebug(f"Matched single word '{target_name}' to object {obj_id}: '{obj_name}'")
            else:
                # Multiple words: check for exact match or if all words are present
                if target_name_lower == obj_name_lower:
                    matched_ids.append(obj_id)
                    rospy.logdebug(f"Exact match '{target_name}' to object {obj_id}: '{obj_name}'")
                elif all(word in obj_name_lower for word in target_words):
                    matched_ids.append(obj_id)
                    rospy.logdebug(f"Partial match '{target_name}' to object {obj_id}: '{obj_name}'")
        
        return matched_ids

class SpatialObjectTracker:
    """Main spatial object tracking node"""
    
    def __init__(self):
        rospy.init_node('spatial_object_tracker', anonymous=True)
        
        # Mode selection: OD (Object Detection) or DENSE_CAPTION
        self.processing_mode = rospy.get_param('~processing_mode', 'OD')
        rospy.loginfo(f"Processing mode: {self.processing_mode}")
        
        # Initialize components
        self.object_registry = DetectedObjectRegistry(
            position_threshold=rospy.get_param('~position_threshold', 1.5),
            confidence_threshold=rospy.get_param('~confidence_threshold', 0.3)
        )
        
        # Pending LLM requests tracking
        self.pending_llm_requests = {}  # object_id -> timestamp
        
        # Shutdown flag for clean exit
        self.shutdown_flag = threading.Event()
        self.publish_timer = None
        
        # Setup service servers
        self.setup_services()
        
        # Subscriber for detected objects (works for both OD and DENSE_CAPTION modes)
        self.detected_objects_sub = rospy.Subscriber('/detected_objects', DetectedObjects, 
                                                   self.detected_objects_callback)
        
        # LLM parser subscribers/publishers (common for both modes)
        self.parser_llm_result_sub = rospy.Subscriber('/scene_graph/parser_llm/result', ObjectSceneGraph, 
                                                   self.parser_llm_result_callback)
        
        # Publisher for graph objects (for graph management node)
        self.graph_objects_pub = rospy.Publisher('/scene_graph/seen_graph_objects', GraphObjects, queue_size=10)
        self.parser_llm_request_pub = rospy.Publisher('/scene_graph/parser_llm', ObjectDescriptions, queue_size=10)
        
        # Publisher for object relationships (for graph management node)
        self.relationships_pub = rospy.Publisher('/scene_graph/object_relationships', String, queue_size=10)

        rospy.loginfo("Spatial Object Tracker initialized")
    
    def setup_services(self):
        """Setup ROS service servers"""
        try:
            rospy.loginfo("Creating register_object service...")
            self.register_service = rospy.Service('/spatial_tracker/register_object', 
                                                RegisterObject, self.handle_register_object)
            rospy.loginfo("✅ register_object service created")
            
            rospy.loginfo("Creating query_objects service...")
            self.query_service = rospy.Service('/spatial_tracker/query_objects', 
                                             QueryObjects, self.handle_query_objects)
            rospy.loginfo("✅ query_objects service created")
            
            rospy.loginfo("Creating get_object_info service...")
            self.info_service = rospy.Service('/spatial_tracker/get_object_info', 
                                            GetObjectInfo, self.handle_get_object_info)
            rospy.loginfo("✅ get_object_info service created")
            
            rospy.loginfo("Creating get_relationships service...")
            self.relationships_service = rospy.Service('/spatial_tracker/get_relationships', 
                                                     QueryObjects, self.handle_get_relationships)
            rospy.loginfo("✅ get_relationships service created")
            
            rospy.loginfo("Service servers initialized")
        except Exception as e:
            rospy.logerr(f"Failed to create services: {e}")
            import traceback
            traceback.print_exc()
    
    def detected_objects_callback(self, msg: DetectedObjects):
        """Process incoming detected objects (handles both OD and DENSE_CAPTION modes)"""
        for detected_obj in msg.objects:
            # Get label
            if hasattr(detected_obj, 'label') and detected_obj.label:
                full_label = detected_obj.label
            else:
                full_label = detected_obj.class_name.data
            
            # Get confidence
            if hasattr(detected_obj, 'confidence') and detected_obj.confidence > 0:
                confidence = detected_obj.confidence
            else:
                confidence = 0.8
            
            # Get position
            if hasattr(detected_obj, 'position'):
                world_position = np.array([
                    detected_obj.position.x,
                    detected_obj.position.y,
                    detected_obj.position.z
                ])
            else:
                # Use default position if not available
                world_position = np.array([0.0, 0.0, 0.0])
            
            # Get size if available
            if hasattr(detected_obj, 'size'):
                object_size = np.array([
                    detected_obj.size.x if detected_obj.size.x > 0 else 0.2,
                    detected_obj.size.y if detected_obj.size.y > 0 else 0.2,
                    detected_obj.size.z if detected_obj.size.z > 0 else 0.2
                ])
            else:
                object_size = np.array([0.2, 0.2, 0.2])  # Default 20cm cube
            
            # Sanitize object size to prevent issues downstream
            object_size = self.object_registry.sanitize_object_size(object_size)
            
            # Handle different processing modes
            if self.processing_mode == 'DENSE_CAPTION':
                # Extract first two words for spatial duplicate detection
                words = full_label.split()
                if len(words) >= 2:
                    basic_label = f"{words[0]} {words[1]}"
                else:
                    basic_label = full_label
                
                rospy.loginfo(f"Processing dense caption: {full_label[:60]}...")
                rospy.logdebug(f"Using basic label for duplicate detection: {basic_label}")
                
                # Check for existing similar objects
                existing_objects = self.object_registry.find_objects_by_label_prefix(basic_label)
                
                if existing_objects:
                    # Found similar objects - check if any need LLM processing
                    for existing_obj_id, obj_instance in existing_objects:
                        if not obj_instance.llm_request_sent and existing_obj_id not in self.pending_llm_requests:
                            # Update position if available, then send to LLM parser
                            if not np.allclose(world_position, [0.0, 0.0, 0.0]):
                                # Update object with new position information
                                self.object_registry.register_or_update_object(
                                    basic_label, world_position, confidence, (0, 0, 100, 100), 
                                    np.array([0.0, 0.0, 0.0]), object_size
                                )
                            
                            # Send to LLM parser
                            self.send_to_llm_parser(existing_obj_id, full_label)
                            obj_instance.llm_request_sent = True
                            obj_instance.raw_description = full_label
                            self.pending_llm_requests[existing_obj_id] = time.time()
                            rospy.loginfo(f"Sent existing object {existing_obj_id} with label '{basic_label}' to LLM parser")
                            break  # Only process one to avoid duplicates
                    else:
                        rospy.logdebug(f"All similar objects already sent to LLM parser: {basic_label}")
                        continue
                else:
                    # No similar objects found - create a new one
                    bbox = (0, 0, 100, 100)  # Default bbox
                    camera_pose = np.array([0.0, 0.0, 0.0])  # Default camera pose
                    confidence = confidence if confidence > 0 else 0.7  # Default confidence for dense captions
                    
                    # Register new object
                    obj_id = self.object_registry.register_or_update_object(
                        basic_label, world_position, confidence, bbox, camera_pose, object_size
                    )
                    
                    # Send to LLM parser immediately
                    obj_instance = self.object_registry.get_object_by_id(obj_id)
                    if obj_instance:
                        self.send_to_llm_parser(obj_id, full_label)
                        obj_instance.llm_request_sent = True
                        obj_instance.raw_description = full_label
                        self.pending_llm_requests[obj_id] = time.time()
                        rospy.loginfo(f"Created new object {obj_id} with label '{basic_label}' and sent to LLM parser")
                        
            else:  # OD mode - original behavior
                # Register object normally
                bbox = (0, 0, 100, 100)  # Default bbox
                camera_pose = np.array([0.0, 0.0, 0.0])  # Default camera pose
                
                obj_id = self.object_registry.register_or_update_object(
                    full_label, world_position, confidence, bbox, camera_pose, object_size
                )
        
        # Publish updated graph objects after processing all detections
        self.publish_graph_objects()
    
    def send_to_llm_parser(self, object_index: int, caption_text: str):
        """Send object to LLM parser for processing"""
        try:
            # Create ObjectDescriptions message with single ObjectDescription
            descriptions_msg = ObjectDescriptions()
            
            # Create single ObjectDescription
            description = ObjectDescription()
            description.object_id = object_index  # Now using correct field name
            description.caption = caption_text    # Now using correct field name
            
            # Add to objects list
            descriptions_msg.objects = [description]
            
            # Publish the request
            self.parser_llm_request_pub.publish(descriptions_msg)
            rospy.logdebug(f"Published LLM parser request for object {object_index}")
            
        except Exception as e:
            rospy.logerr(f"Error sending object {object_index} to LLM parser: {e}")
    
    def parser_llm_result_callback(self, msg: ObjectSceneGraph):
        """Handle LLM parser results"""
        try:
            # Get object_index from the main_object.id field
            object_index = msg.main_object.id
            
            # Validate that we have a valid object index
            if object_index <= 0:
                rospy.logwarn(f"Invalid object_index received: {object_index}")
                return
            
            # Remove from pending requests
            self.pending_llm_requests.pop(object_index, None)
            
            # Get the object instance
            obj_instance = self.object_registry.get_object_by_id(object_index)
            if not obj_instance:
                rospy.logwarn(f"Object {object_index} not found in registry")
                return
            
            # Convert ObjectSceneGraph message to dictionary format for backward compatibility
            parsed_data = {
                'main_object': {
                    'name': msg.main_object.name,
                    'attributes': {
                        'color': msg.main_object.attributes.color,
                        'material': msg.main_object.attributes.material,
                        'style': msg.main_object.attributes.style
                    }
                },
                'parts': [
                    {
                        'name': part.name,
                        'attributes': {
                            'color': part.attributes.color,
                            'material': part.attributes.material,
                            'style': part.attributes.style
                        },
                        'relationship_to_main': part.relationship_to_main
                    }
                    for part in msg.parts
                ],
                'environment': [
                    {
                        'name': env.name,
                        'attributes': {
                            'color': env.attributes.color,
                            'material': env.attributes.material,
                            'style': env.attributes.style
                        }
                    }
                    for env in msg.environment
                ],
                'spatial_context': {
                    'position': msg.spatial_context.position,
                    'nearby_objects': list(msg.spatial_context.nearby_objects)
                }
            }
            
            # Store the parsed results
            obj_instance.llm_parsed = True
            obj_instance.parsed_json = parsed_data
            obj_instance.main_object = parsed_data['main_object']
            obj_instance.parts = parsed_data['parts']
            obj_instance.environment = parsed_data['environment']
            obj_instance.spatial_context = parsed_data['spatial_context']
            
            rospy.loginfo(f"Updated object {object_index} with LLM parsing results")
            rospy.logdebug(f"Main object: {obj_instance.main_object}")
            
            # Update object relationships using KD-tree
            self.update_object_relationships(object_index)
            
            rospy.loginfo(f"Object {object_index} relationship mapping complete")
            
        except Exception as e:
            rospy.logerr(f"Error processing LLM parser result: {e}")
            import traceback
            traceback.print_exc()
    
    def update_object_relationships(self, object_id: int, search_radius: float = 3.0):
        """Update nearby objects and mapped relationships for a given object"""
        obj_instance = self.object_registry.get_object_by_id(object_id)
        if not obj_instance or not obj_instance.llm_parsed:
            rospy.logdebug(f"Object {object_id} not found or not LLM parsed")
            return
        
        # Find nearby objects using KD-tree
        nearby_objects = self.object_registry.find_nearby_objects(object_id, search_radius)
        obj_instance.nearby_objects = nearby_objects
        
        rospy.loginfo(f"Found {len(nearby_objects)} nearby objects for object {object_id}")
        
        # Clear existing mapped relationships
        obj_instance.mapped_relationships.clear()
        
        # Map parts to nearby objects
        if obj_instance.parts:
            for part in obj_instance.parts:
                part_name = part.get('name', '')
                if part_name:
                    matched_ids = self.object_registry.map_name_to_objects(part_name, nearby_objects)
                    for matched_id in matched_ids:
                        relationship = part.get('relationship_to_main', 'part_of')
                        obj_instance.mapped_relationships[matched_id] = relationship
                        rospy.loginfo(f"Mapped part '{part_name}' to object {matched_id} with relationship '{relationship}'")
        
        # Map spatial context objects
        if obj_instance.spatial_context:
            nearby_object_names = obj_instance.spatial_context.get('nearby_objects', [])
            spatial_position = obj_instance.spatial_context.get('position', 'nearby')
            
            for nearby_name in nearby_object_names:
                if nearby_name:
                    matched_ids = self.object_registry.map_name_to_objects(nearby_name, nearby_objects)
                    for matched_id in matched_ids:
                        # Don't overwrite part relationships
                        if matched_id not in obj_instance.mapped_relationships:
                            obj_instance.mapped_relationships[matched_id] = spatial_position
                            rospy.loginfo(f"Mapped spatial context '{nearby_name}' to object {matched_id} with position '{spatial_position}'")
        
        rospy.loginfo(f"Object {object_id} now has {len(obj_instance.mapped_relationships)} mapped relationships")
    
    def get_all_object_relationships(self) -> Dict[int, Dict[int, str]]:
        """Get all object relationships for the entire registry"""
        all_relationships = {}
        
        with self.object_registry.lock:
            for obj_id, obj_instance in self.object_registry.objects.items():
                if obj_instance.mapped_relationships:
                    all_relationships[obj_id] = obj_instance.mapped_relationships.copy()
        
        return all_relationships
    
    def get_object_relationships(self, object_id: int) -> Dict[int, str]:
        """Get mapped relationships for a specific object"""
        obj_instance = self.object_registry.get_object_by_id(object_id)
        if obj_instance:
            return obj_instance.mapped_relationships.copy()
        return {}
    
    def publish_graph_objects(self):
        """Publish all tracked objects as GraphObjects for the graph management node"""
        try:
            graph_objects_msg = GraphObjects()
            graph_objects_msg.header.stamp = rospy.Time.now()
            
            graph_objects = []
            
            with self.object_registry.lock:
                object_count = len(self.object_registry.objects)
                rospy.loginfo(f"Publishing {object_count} objects as GraphObjects")
                
                for obj_id, obj_instance in self.object_registry.objects.items():
                    # Create GraphObject
                    graph_object = GraphObject()
                    
                    # Set object name/label
                    graph_object.name = String(data=obj_instance.label)
                    
                    # Set object ID from spatial tracker
                    graph_object.object_id = Int32(data=obj_id)
                    
                    # Set image index (use object ID if no specific image index available)
                    #graph_object.image_index = Int32(data=obj_id)
                    
                    # Set description
                    #graph_object.description = String(data=f"Tracked object: {obj_instance.label} (ID: {obj_id})")
                    
                    # Create 3D bounding box from position and actual object size
                    bounding_box_3d = self.object_registry.create_3d_bounding_box(obj_instance.world_position, obj_instance.object_size)
                    graph_object.bounding_box = bounding_box_3d
                    
                    graph_objects.append(graph_object)
                    
                    #rospy.logdebug(f"Created GraphObject for {obj_instance.label} at ({obj_instance.world_position[0]:.2f}, {obj_instance.world_position[1]:.2f}, {obj_instance.world_position[2]:.2f})")
            
            graph_objects_msg.objects = graph_objects
            
            # Publish the graph objects
            self.graph_objects_pub.publish(graph_objects_msg)
            
            # Also publish relationships as JSON string
            all_relationships = self.get_all_object_relationships()
            if all_relationships:
                # Convert numpy types to native Python types for JSON serialization
                json_safe_relationships = {}
                for obj_id, relationships in all_relationships.items():
                    # Convert main object ID to int
                    json_obj_id = int(obj_id)
                    json_safe_relationships[json_obj_id] = {}
                    
                    # Convert relationship target IDs to int
                    for related_id, relationship_desc in relationships.items():
                        json_related_id = int(related_id)
                        json_safe_relationships[json_obj_id][json_related_id] = str(relationship_desc)
                
                relationships_json = json.dumps(json_safe_relationships)
                relationships_msg = String()
                relationships_msg.data = relationships_json
                self.relationships_pub.publish(relationships_msg)
                rospy.logdebug(f"Published relationships for {len(json_safe_relationships)} objects")
            
            rospy.loginfo(f"Published {len(graph_objects)} GraphObjects to /scene_graph/seen_graph_objects")
            
        except Exception as e:
            rospy.logerr(f"Error publishing graph objects: {e}")
            import traceback
            traceback.print_exc()

    def handle_register_object(self, req):
        """Handle object registration service request"""
        try:
            rospy.loginfo("Received register object service request")
            detected_obj = req.object
            
            # Get label
            if hasattr(detected_obj, 'label') and detected_obj.label:
                label = detected_obj.label
            else:
                label = detected_obj.class_name.data
            
            # Get confidence
            if hasattr(detected_obj, 'confidence') and detected_obj.confidence > 0:
                confidence = detected_obj.confidence
            else:
                confidence = 0.8
            
            # Get position
            if hasattr(detected_obj, 'position'):
                world_position = np.array([
                    detected_obj.position.x,
                    detected_obj.position.y,
                    detected_obj.position.z
                ])
            else:
                world_position = np.array([0.0, 0.0, 0.0])
            
            # Get size if available
            if hasattr(detected_obj, 'size'):
                object_size = np.array([
                    detected_obj.size.x if detected_obj.size.x > 0 else 0.2,
                    detected_obj.size.y if detected_obj.size.y > 0 else 0.2,
                    detected_obj.size.z if detected_obj.size.z > 0 else 0.2
                ])
            else:
                object_size = np.array([0.2, 0.2, 0.2])
            
            # Sanitize object size to prevent issues downstream
            object_size = self.object_registry.sanitize_object_size(object_size)
            
            # Register object
            bbox = (0, 0, 100, 100)
            camera_pose = np.array([0.0, 0.0, 0.0])
            
            obj_id = self.object_registry.register_or_update_object(
                label, world_position, confidence, bbox, camera_pose, object_size
            )
            
            response = RegisterObjectResponse()
            response.success = True
            response.object_id = str(obj_id)
            response.message = f"Successfully registered {label} with ID {obj_id}"
            
            return response
            
        except Exception as e:
            rospy.logerr(f"Error registering object: {e}")
            response = RegisterObjectResponse()
            response.success = False
            response.object_id = ""
            response.message = f"Failed to register object: {str(e)}"
            return response
    
    def handle_query_objects(self, req):
        """Handle object query service request"""
        try:
            center = np.array([req.center.x, req.center.y, req.center.z])
            radius = req.radius
            label_filter = req.label_filter if req.label_filter else None
            
            # Query objects
            if label_filter:
                object_ids = self.object_registry.query_objects_by_label(label_filter)
                # Filter by distance
                filtered_ids = []
                for obj_id in object_ids:
                    obj_instance = self.object_registry.objects.get(obj_id)
                    if obj_instance:
                        distance = np.linalg.norm(obj_instance.world_position - center)
                        if distance <= radius:
                            filtered_ids.append(obj_id)
                object_ids = filtered_ids
            else:
                object_ids = self.object_registry.query_objects_spatial(center, radius)
            
            response = QueryObjectsResponse()
            response.object_ids = [str(obj_id) for obj_id in object_ids]
            
            return response
            
        except Exception as e:
            rospy.logerr(f"Error querying objects: {e}")
            response = QueryObjectsResponse()
            response.object_ids = []
            return response
    
    def handle_get_object_info(self, req):
        """Handle get object info service request"""
        try:
            obj_id = int(req.object_id)
            obj_instance = self.object_registry.objects.get(obj_id)
            
            if obj_instance is None:
                response = GetObjectInfoResponse()
                response.success = False
                response.message = f"Object with ID {obj_id} not found"
                return response
            
            # Create DetectedObject message
            detected_obj = DetectedObject()
            
            # Populate old format fields
            detected_obj.class_name = String(data=obj_instance.label)
            detected_obj.image_index = Int32(data=obj_id)
            detected_obj.description = String(data=f"ID:{obj_id} {obj_instance.label}")
            
            # New format fields
            if hasattr(detected_obj, 'label'):
                detected_obj.label = obj_instance.label
            if hasattr(detected_obj, 'confidence'):
                detected_obj.confidence = obj_instance.confidence
            if hasattr(detected_obj, 'position'):
                detected_obj.position.x = obj_instance.world_position[0]
                detected_obj.position.y = obj_instance.world_position[1]
                detected_obj.position.z = obj_instance.world_position[2]
            
            response = GetObjectInfoResponse()
            response.success = True
            response.object = detected_obj
            response.message = f"Found object {obj_instance.label} with ID {obj_id}"
            
            return response
            
        except Exception as e:
            rospy.logerr(f"Error getting object info: {e}")
            response = GetObjectInfoResponse()
            response.success = False
            response.message = f"Error retrieving object: {str(e)}"
            return response
    
    def handle_get_relationships(self, req):
        """Handle get relationships service request - returns JSON string of all relationships"""
        try:
            all_relationships = self.get_all_object_relationships()
            
            # Convert to JSON string for easy parsing by graph management node
            relationships_json = json.dumps(all_relationships)
            
            response = QueryObjectsResponse()
            response.object_ids = [relationships_json]  # Use object_ids field to return JSON string
            
            rospy.loginfo(f"Returning relationships for {len(all_relationships)} objects")
            return response
            
        except Exception as e:
            rospy.logerr(f"Error getting relationships: {e}")
            response = QueryObjectsResponse()
            response.object_ids = []
            return response
    
    def shutdown_hook(self):
        """Clean shutdown handler"""
        rospy.loginfo("Shutting down Spatial Object Tracker...")
        
        # Set shutdown flag
        self.shutdown_flag.set()
        
        # Clean up timer
        if self.publish_timer is not None:
            self.publish_timer.shutdown()
            rospy.loginfo("Timer shutdown complete")
        
        # Clean up subscribers and publishers
        try:
            if hasattr(self, 'detected_objects_sub'):
                self.detected_objects_sub.unregister()
            if hasattr(self, 'parser_llm_result_sub'):
                self.parser_llm_result_sub.unregister()
            if hasattr(self, 'graph_objects_pub'):
                self.graph_objects_pub.unregister()
            if hasattr(self, 'parser_llm_request_pub'):
                self.parser_llm_request_pub.unregister()
            if hasattr(self, 'relationships_pub'):
                self.relationships_pub.unregister()
            rospy.loginfo("ROS subscribers/publishers cleaned up")
        except Exception as e:
            rospy.logwarn(f"Error during ROS cleanup: {e}")
        
        # Clear pending requests
        self.pending_llm_requests.clear()
        
        # Clean up object registry
        if hasattr(self, 'object_registry'):
            self.object_registry.cleanup()
        
        rospy.loginfo("Spatial Object Tracker shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle system signals for clean shutdown"""
        rospy.loginfo(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_hook()
        sys.exit(0)
    
    def run(self):
        """Main execution loop"""
        # Register shutdown handlers
        rospy.on_shutdown(self.shutdown_hook)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        rospy.loginfo("Spatial Object Tracker running...")
        rospy.loginfo(f"Position threshold: {self.object_registry.position_threshold}m")
        rospy.loginfo("Publishing tracked objects to /scene_graph/seen_graph_objects")
        
        try:
            # Set up periodic publishing timer (publish every 2 seconds)
            self.publish_timer = rospy.Timer(rospy.Duration(2.0), self.periodic_publish_callback)
            
            # Spin until shutdown
            while not rospy.is_shutdown() and not self.shutdown_flag.is_set():
                rospy.sleep(0.1)
                
        except KeyboardInterrupt:
            rospy.loginfo("KeyboardInterrupt received")
            self.shutdown_hook()
        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            self.shutdown_hook()
        finally:
            # Ensure cleanup happens
            if self.publish_timer is not None:
                self.publish_timer.shutdown()
    
    def periodic_publish_callback(self, event):
        """Periodic callback to publish graph objects"""
        # Check if shutdown has been requested
        if self.shutdown_flag.is_set():
            return
            
        object_count = self.object_registry.get_object_count()
        rospy.loginfo(f"Timer callback: {object_count} objects in registry")
        if object_count > 0:
            rospy.loginfo("Publishing graph objects from timer")
            self.publish_graph_objects()
        else:
            rospy.loginfo("No objects to publish")

if __name__ == '__main__':
    try:
        tracker = SpatialObjectTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Spatial Object Tracker interrupted")
    except Exception as e:
        rospy.logerr(f"Error in Spatial Object Tracker: {e}")
        import traceback
        traceback.print_exc()
