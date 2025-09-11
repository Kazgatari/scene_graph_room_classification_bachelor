#!/usr/bin/env python3
from cProfile import label

import numpy as np
import rospy
from torch import device
import torch
#from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForCausalLM
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import cv2
import time
from scene_graph.srv import Florence2, Florence2Response
import sys
from scene_graph.msg import DetectedObjects, DetectedObject
from geometry_msgs.msg import Point32, PointStamped, TransformStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
import struct
# import tf2_ros  # Commented out due to PyKDL dependency issues
# import tf2_geometry_msgs  # Commented out due to PyKDL dependency issues
import message_filters
# for Json comparison
import json
import pickle
import os
from datetime import datetime
from collections import Counter
import signal

class Florence2Node:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_callback_time = 0  # Track the last callback time
        self.callback_interval = 1.0  # Interval in seconds (e.g., 1 Hz)
        self.detected_objects_list = []
        
        # Collection to store all detected objects
        self.detected_objects_collection = []
        self.output_folder = '/root/catkin_ws/src/scene_graph_room_classification/detection_results'
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Check GPU compatibility first
        if torch.cuda.is_available():
            try:
                # Test GPU compatibility with a simple operation
                test_tensor = torch.tensor([1.0]).cuda()
                test_result = test_tensor + 1.0
                self.device = "cuda"
                self.torch_dtype = torch.float16
                rospy.loginfo("CUDA available and compatible, using GPU")
            except Exception as e:
                rospy.logwarn(f"CUDA available but incompatible ({e}), falling back to CPU")
                self.device = "cpu"
                self.torch_dtype = torch.float32
        else:
            rospy.loginfo("CUDA not available, using CPU")
            self.device = "cpu"
            self.torch_dtype = torch.float32
            
        rospy.loginfo(f"Loading Florence-2 model on {self.device} with dtype {self.torch_dtype}")
        
        # Set environment variable to avoid trust_remote_code prompts
        os.environ['HF_HUB_DISABLE_INTERACTIVE'] = '1'
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                "microsoft/Florence-2-large-ft", 
                trust_remote_code=True,
                cache_dir="/tmp/huggingface_cache"  # Optional: specify cache directory
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large-ft", 
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                cache_dir="/tmp/huggingface_cache",  # Optional: specify cache directory
                attn_implementation="eager"  # Force eager attention to avoid _supports_sdpa issues
            )
            
            self.model = self.model.to(self.device, dtype=self.torch_dtype)
            rospy.loginfo("Florence-2 model loaded successfully")
            
        except Exception as e:
            rospy.logerr(f"Failed to load Florence-2 model: {e}")
            raise e
        
        self.image_path = '/root/catkin_ws/src/scene_graph_room_classification/images/detected_objects/'
        self.service = rospy.Service('florence2_service', Florence2, self.handle_service)
        # Subscribe to the image topic
        #rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/points', PointCloud2)
        self.odom_sub = message_filters.Subscriber('/odom', Odometry)

        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.odom_sub], queue_size=100, slop=0.1)
        ts.registerCallback(self.synchronized_callback)

        self.image_pub = rospy.Publisher('/scene_graph/color/image_raw', Image, queue_size=1)
        #self.segmented_image_pub = rospy.Publisher('/camera/color/segmented_image', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/scene_graph/depth/points', PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher('/scene_graph/odom', Odometry, queue_size=1)
        self.detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjects, queue_size=10)  # Changed to /detected_objects for spatial tracker

        # Add camera info subscriber for 3D coordinate calculation
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        self.camera_info = None
        
        # TF2 transforms commented out due to PyKDL dependency issues
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Default camera intrinsics (will be updated from camera_info)
        self.fx = 615.0  # focal length x
        self.fy = 615.0  # focal length y
        self.cx = 320.0  # principal point x
        self.cy = 240.0  # principal point y
        
        rospy.loginfo("Florence-2 node initialized with 3D coordinate conversion support")

    def letterbox_image(self, image, target_size=768):
        """
        Apply letterboxing to resize image to target_size x target_size while maintaining aspect ratio.
        Returns the letterboxed image and scaling factors for coordinate conversion.
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor to fit the image within target_size
        scale = min(target_size / w, target_size / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a blank canvas of target_size x target_size
        letterboxed = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # 114 is a neutral gray
        
        # Calculate padding offsets to center the image
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        
        # Place the resized image in the center of the canvas
        letterboxed[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Return letterboxed image and transformation parameters
        return letterboxed, {
            'scale': scale,
            'pad_x': pad_x, 
            'pad_y': pad_y,
            'original_size': (w, h),
            'letterboxed_size': (target_size, target_size),
            'resized_size': (new_w, new_h)
        }
    
    def convert_coordinates_back(self, bboxes, transform_params):
        """
        Convert bounding box coordinates from letterboxed image back to original image coordinates.
        """
        scale = transform_params['scale']
        pad_x = transform_params['pad_x']
        pad_y = transform_params['pad_y']
        orig_w, orig_h = transform_params['original_size']
        
        converted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Remove padding offset
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = max(0, x2 - pad_x)
            y2 = max(0, y2 - pad_y)
            
            # Scale back to original size
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
            
            # Ensure coordinates are within original image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            converted_bboxes.append([x1, y1, x2, y2])
        
        return converted_bboxes

    def camera_info_callback(self, msg):
        """Store camera intrinsics for 3D coordinate calculation"""
        self.camera_info = msg
        self.fx = msg.K[0]  # K[0] = fx
        self.fy = msg.K[4]  # K[4] = fy
        self.cx = msg.K[2]  # K[2] = cx
        self.cy = msg.K[5]  # K[5] = cy
        rospy.loginfo_once(f"Camera intrinsics updated: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
    
    def sample_depth_in_bbox(self, bbox, depth_msg):
        """Sample depth values within bounding box from point cloud"""
        x1, y1, x2, y2 = bbox
        
        # Convert to integers and ensure within image bounds
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        
        depth_values = []
        
        try:
            # Parse point cloud data
            point_step = depth_msg.point_step
            row_step = depth_msg.row_step
            
            # Sample points within bounding box
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # Sample every few pixels to get representative depth
            step_x = max(1, bbox_width // 10)
            step_y = max(1, bbox_height // 10)
            
            for y in range(y1, y2, step_y):
                for x in range(x1, x2, step_x):
                    # Calculate byte offset in point cloud
                    array_position = y * row_step + x * point_step
                    
                    if array_position + 12 <= len(depth_msg.data):  # Ensure we have enough data
                        # Extract XYZ coordinates (assuming PointXYZ format)
                        x_bytes = depth_msg.data[array_position:array_position+4]
                        y_bytes = depth_msg.data[array_position+4:array_position+8]
                        z_bytes = depth_msg.data[array_position+8:array_position+12]
                        
                        # Unpack as float32
                        x_val = struct.unpack('f', x_bytes)[0]
                        y_val = struct.unpack('f', y_bytes)[0]
                        z_val = struct.unpack('f', z_bytes)[0]
                        
                        # Use Z coordinate as depth (distance from camera)
                        if not (np.isnan(z_val) or np.isinf(z_val)) and z_val > 0.1:  # Valid depth > 10cm
                            depth_values.append(z_val)
            
            return depth_values if depth_values else [2.0]  # Default 2m if no valid depth
            
        except Exception as e:
            rospy.logwarn(f"Error sampling depth: {e}")
            return [2.0]  # Default depth
    
    def pixel_to_camera_coords(self, pixel_x, pixel_y, depth):
        """Convert pixel coordinates + depth to 3D camera coordinates"""
        # Convert from image coordinates to camera coordinates
        camera_x = (pixel_x - self.cx) * depth / self.fx
        camera_y = (pixel_y - self.cy) * depth / self.fy
        camera_z = depth
        
        return np.array([camera_x, camera_y, camera_z])
    
    def transform_to_world_coords(self, camera_point, odom_msg=None):
        """Transform point from camera frame to world frame using odometry"""
        try:
            if odom_msg is None:
                # No odometry available, return camera coordinates
                rospy.logwarn_once("No odometry data available, using camera coordinates")
                return camera_point
            
            # Get robot pose from odometry
            robot_x = odom_msg.pose.pose.position.x
            robot_y = odom_msg.pose.pose.position.y
            robot_z = odom_msg.pose.pose.position.z
            
            # Get robot orientation (quaternion)
            qx = odom_msg.pose.pose.orientation.x
            qy = odom_msg.pose.pose.orientation.y
            qz = odom_msg.pose.pose.orientation.z
            qw = odom_msg.pose.pose.orientation.w
            
            # Convert quaternion to rotation matrix (simplified for yaw rotation)
            # For a more complete solution, you'd use full 3D rotation
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            # Simple 2D transformation (assumes camera points forward)
            # Rotate camera coordinates by robot yaw
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Transform from camera frame to robot frame (assuming camera faces forward)
            # Camera X -> forward, Camera Y -> left, Camera Z -> up
            robot_relative_x = camera_point[2]  # Camera Z becomes robot X (forward)
            robot_relative_y = -camera_point[0]  # Camera -X becomes robot Y (left)
            robot_relative_z = -camera_point[1]  # Camera -Y becomes robot Z (up)
            
            # Rotate by robot orientation and translate by robot position
            world_x = robot_x + (robot_relative_x * cos_yaw - robot_relative_y * sin_yaw)
            world_y = robot_y + (robot_relative_x * sin_yaw + robot_relative_y * cos_yaw)
            world_z = robot_z + robot_relative_z
            
            return np.array([world_x, world_y, world_z])
            
        except Exception as e:
            rospy.logwarn(f"Coordinate transform failed: {e}, using camera coordinates")
            return camera_point
    
    def calculate_3d_world_position(self, bbox, depth_msg, odom_msg=None):
        """Calculate 3D world position from 2D bounding box and depth data"""
        x1, y1, x2, y2 = bbox
        
        # Get center pixel of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Sample depth values in bounding box region
        depth_samples = self.sample_depth_in_bbox(bbox, depth_msg)
        
        # Use median depth for robustness
        estimated_depth = np.median(depth_samples)
        
        # rospy.logdebug(f"Bbox center: ({center_x:.1f}, {center_y:.1f}), depth samples: {len(depth_samples)}, median depth: {estimated_depth:.2f}m")
        
        # Convert to 3D camera coordinates
        camera_point = self.pixel_to_camera_coords(center_x, center_y, estimated_depth)
        
        # Transform to world coordinates using odometry
        world_point = self.transform_to_world_coords(camera_point, odom_msg)
        
        return world_point

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        rospy.loginfo(f"Received signal {signum}, shutting down gracefully...")
        self.save_collection_and_exit()
        rospy.signal_shutdown("User requested shutdown")
        
    def save_collection_and_exit(self):
        """Save the detected objects collection and statistics, then exit"""
        if not self.detected_objects_collection:
            rospy.loginfo("No objects detected, nothing to save.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create class count dictionary
        class_counts = self.create_class_count_dictionary()
        
        # Save as JSON (human-readable)
        json_file = os.path.join(self.output_folder, f"detected_objects_{timestamp}.json")
        counts_file = os.path.join(self.output_folder, f"class_counts_{timestamp}.json")
        
        try:
            # Save full collection
            with open(json_file, 'w') as f:
                json.dump(self.detected_objects_collection, f, indent=2, default=str)
            
            # Save class counts
            with open(counts_file, 'w') as f:
                json.dump(class_counts, f, indent=2)
            
            # Also save as pickle for faster loading if needed
            #pickle_file = os.path.join(self.output_folder, f"detected_objects_{timestamp}.pkl")
            #with open(pickle_file, 'wb') as f:
                #pickle.dump(self.detected_objects_collection, f)
            
            # Save summary as text file
            txt_file = os.path.join(self.output_folder, f"detection_summary_{timestamp}.txt")
            with open(txt_file, 'w') as f:
                f.write(f"Detection Summary - {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total objects detected: {len(self.detected_objects_collection)}\n")
                f.write(f"Unique classes: {len(class_counts)}\n\n")
                f.write("Class distribution:\n")
                for class_name, count in sorted(class_counts.items()):
                    f.write(f"  {class_name}: {count}\n")
                #f.write(f"\nData saved to:\n")
                #f.write(f"  - JSON: {json_file}\n")
                #f.write(f"  - Pickle: {pickle_file}\n")
                #f.write(f"  - Counts: {counts_file}\n")
            
            rospy.loginfo(f"Saved {len(self.detected_objects_collection)} detected objects to {self.output_folder}")
            rospy.loginfo(f"Class counts: {class_counts}")
            
        except Exception as e:
            rospy.logerr(f"Error saving collection: {e}")
        
        # Exit gracefully
        rospy.signal_shutdown("User requested shutdown")
        
    def create_class_count_dictionary(self):
        """Create a dictionary with counts of each class"""
        class_names = [obj['class_name'] for obj in self.detected_objects_collection]
        return dict(Counter(class_names))

    def synchronized_callback(self, image_msg, depth_msg, odom_msg):
        detected_objects = []
        indices = []
        i = 0

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')

        # Use object detection for spatial tracking
        results = self.run_general(task_prompt="<DENSE_REGION_CAPTION>", cv_image=cv_image) # Use <OD> for spatial tracking else <DENSE_REGION_CAPTION>
        if results is None:
            rospy.logwarn("No detection results received")
            return
        
        bboxes = results['bboxes']
        labels = results['labels']
        
        # rospy.loginfo(f"Florence-2 detected {len(labels)} objects: {labels}")
        
        # Store detection timestamp
        detection_timestamp = rospy.Time.now().to_sec()
        
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            
            # Calculate 3D world position from bbox and depth data
            world_position = self.calculate_3d_world_position(bbox, depth_msg, odom_msg)
            
            detected_object = DetectedObject()
            
            # Original format fields
            detected_object.class_name = String(data=label)
            detected_object.image_index = Int32(data=i)
            detected_object.bounding_box = [
                Point32(x=x1, y=y1, z=0),
                Point32(x=x2, y=y2, z=0)
            ]
            detected_object.segment = []  # Empty for now
            detected_object.description = String(data=label)
            
            # New format fields for spatial tracking with REAL 3D world coordinates
            if hasattr(detected_object, 'label'):
                detected_object.label = label
            if hasattr(detected_object, 'confidence'):
                detected_object.confidence = 0.9  # Default confidence for Florence-2 detections
            if hasattr(detected_object, 'position'):
                # NOW using actual 3D world coordinates!
                detected_object.position.x = world_position[0]  # meters in world frame
                detected_object.position.y = world_position[1]  # meters in world frame  
                detected_object.position.z = world_position[2]  # meters in world frame
            if hasattr(detected_object, 'size'):
                # Estimate physical size from bounding box and depth
                depth = world_position[2] if len(world_position) > 2 else 2.0
                width_pixels = x2 - x1
                height_pixels = y2 - y1
                # Convert pixel size to physical size using camera intrinsics
                width_meters = (width_pixels * depth) / self.fx
                height_meters = (height_pixels * depth) / self.fy
                detected_object.size.x = width_meters
                detected_object.size.y = height_meters
                detected_object.size.z = min(width_meters, height_meters)  # Estimated depth
            if hasattr(detected_object, 'bbox'):
                detected_object.bbox = [int(x1), int(y1), int(x2), int(y2)]

            detected_objects.append(detected_object)
            i += 1
            
            # Add to collection for saving later with world coordinates
            detection_data = {
                'timestamp': detection_timestamp,
                'class_name': label,
                'bounding_box': {
                    'x1': float(x1),
                    'y1': float(y1), 
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'world_position': {
                    'x': float(world_position[0]),
                    'y': float(world_position[1]),
                    'z': float(world_position[2])
                },
                'image_dimensions': {
                    'width': cv_image.shape[1],
                    'height': cv_image.shape[0]
                }
            }
            self.detected_objects_collection.append(detection_data)
            
            #rospy.loginfo(f"Detected {label} at world position ({world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f})")

        # Create a DetectedObjects message
        detected_objects_msg = DetectedObjects()
        detected_objects_msg.objects = detected_objects
        detected_objects_msg.header.stamp = rospy.Time.now()
        depth_msg.header.stamp = rospy.Time.now()
        image_msg.header.stamp = rospy.Time.now()
        odom_msg.header.stamp = rospy.Time.now()
        
        image_msg.header.frame_id = 'map'
        depth_msg.header.frame_id = 'map'
        odom_msg.header.frame_id = 'map'

        self.image_pub.publish(image_msg)
        self.depth_pub.publish(depth_msg)
        self.odom_pub.publish(odom_msg)
        self.detected_objects_pub.publish(detected_objects_msg)

    def run_general(self, task_prompt, text_input=None, cv_image=None):
        if cv_image is None:
            return
        
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        # Store original dimensions
        original_height, original_width = cv_image.shape[:2]
        #print(f"Original image dimensions: {original_width}x{original_height}")

        # Apply letterboxing for better model performance
        letterboxed_image, transform_params = self.letterbox_image(cv_image, target_size=768)
        letterbox_height, letterbox_width = letterboxed_image.shape[:2]
        #print(f"Letterboxed image dimensions: {letterbox_width}x{letterbox_height}")

        # Process the letterboxed image
        inputs = self.processor(text=prompt, images=letterboxed_image, return_tensors="pt").to(self.device, self.torch_dtype)
        
        # Adjust generation parameters based on device
        max_tokens = 512 if self.device == "cpu" else 1024
        num_beams = 1  # Force greedy decoding to avoid cache issues
        
        generated_ids = self.model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=max_tokens,
          num_beams=num_beams,
          early_stopping=False,  # Disable early stopping for greedy decoding
          do_sample=False,       # Ensure deterministic output
          use_cache=False        # Disable cache to avoid cache layer issues
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse results using letterboxed image size
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(letterbox_width, letterbox_height))

        if task_prompt == "<OD>":
            od_result = parsed_answer['<OD>']
            
            # Convert bounding boxes back to original image coordinates
            if 'bboxes' in od_result and od_result['bboxes']:
                original_bboxes = self.convert_coordinates_back(od_result['bboxes'], transform_params)
                od_result['bboxes'] = original_bboxes
                """
                # Debug: Print bounding boxes in original coordinates
                print(f"Detected {len(od_result['bboxes'])} objects:")
                for i, (bbox, label) in enumerate(zip(od_result['bboxes'], od_result['labels'])):
                    x1, y1, x2, y2 = bbox
                    print(f"  Object {i}: {label} at ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                    
                    # Check if coordinates are within original image bounds
                    if x1 < 0 or y1 < 0 or x2 > original_width or y2 > original_height:
                        print(f"    WARNING: Coordinates outside original image bounds!")
                """
            
            return od_result
        
        elif task_prompt == "<DENSE_REGION_CAPTION>":
            od_result = parsed_answer['<DENSE_REGION_CAPTION>']
            
            # Convert bounding boxes back to original image coordinates for dense region captions too
            if 'bboxes' in od_result and od_result['bboxes']:
                original_bboxes = self.convert_coordinates_back(od_result['bboxes'], transform_params)
                od_result['bboxes'] = original_bboxes
                
                """
                print(f"Detected {len(od_result['bboxes'])} regions with captions:")
                for i, (bbox, label) in enumerate(zip(od_result['bboxes'], od_result['labels'])):
                    x1, y1, x2, y2 = bbox
                    print(f"  Region {i}: '{label}' at ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
                """
            
            return od_result
        
        return None
    
    def get_segmentation_mask(self, cv_image, bbox, label):
        try:
            x1, y1, x2, y2 = map(int, bbox)

            # Crop the region of interest
            roi = cv_image[y1:y2, x1:x2]

            # Use Florence-2 segmentation task
            task = "<REFERRING_EXPRESSION_SEGMENTATION>"
            prompt = f"<REFERRING_EXPRESSION_SEGMENTATION>{label}"
            inputs = self.processor(images=roi, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)

            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,  # Reduced from 1024 for faster processing
                num_beams=1,         # Force greedy decoding to avoid cache issues
                do_sample=False,     # Deterministic output
                early_stopping=False,
                use_cache=False      # Disable cache to avoid cache layer issues
            )

            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
            
            
            # Parse the segmentation result
            roi_height, roi_width = roi.shape[:2]
            parsed_result = self.processor.post_process_generation(
                generated_text, 
                task=task, 
                image_size=(roi_width, roi_height)
            )
            polygon_points = []
            print(f"DEBUG: Parsed result for {label}: {parsed_result}")
            # Extract polygon points from the result
            if '<REFERRING_EXPRESSION_SEGMENTATION>' in parsed_result:
                seg_result = parsed_result['<REFERRING_EXPRESSION_SEGMENTATION>']

                if 'polygons' in seg_result and seg_result['polygons']:
                    print(f"DEBUG: Found {len(seg_result['polygons'])} polygon groups for {label}")

                    # Process each polygon group (Florence-2 returns nested arrays)
                    for polygon_group_idx, polygon_group in enumerate(seg_result['polygons']):
                        print(f"DEBUG: Processing polygon group {polygon_group_idx} with {len(polygon_group)} polygons")

                        # Each polygon_group contains multiple polygons
                        for polygon_idx, polygon in enumerate(polygon_group):
                            print(f"DEBUG: Processing polygon {polygon_idx} with {len(polygon)} coordinate values")

                            # Process each coordinate pair in the polygon
                            # Note: polygon is a flat list of [x1, y1, x2, y2, x3, y3, ...]
                            for i in range(0, len(polygon), 2):
                                if i + 1 < len(polygon):  # Make sure we have both x and y
                                    roi_x = polygon[i]
                                    roi_y = polygon[i + 1]

                                    # Convert from ROI coordinates to full image coordinates
                                    abs_x = int(roi_x + x1)
                                    abs_y = int(roi_y + y1)

                                    # Ensure coordinates are within image bounds
                                    abs_x = max(0, min(abs_x, cv_image.shape[1] - 1))
                                    abs_y = max(0, min(abs_y, cv_image.shape[0] - 1))

                                    polygon_points.append(Point32(x=abs_x, y=abs_y, z=0.0))

                    if polygon_points:
                        print(f"INFO: Created {len(polygon_points)} polygon points for {label}")
                        return polygon_points
                    else:
                        print(f"WARNING: No valid polygon points created for {label}")

            # Fallback to rectangular segment if segmentation fails
            print(f"INFO: Using rectangular segment for {label}")
            return [
                Point32(x=x1, y=y1, z=0), # Top-left
                Point32(x=x1, y=y2, z=0), # Bottom-left
                Point32(x=x2, y=y2, z=0), # Bottom-right
                Point32(x=x2, y=y1, z=0)  # Top-right
            ]

        except Exception as e:
            print(f"ERROR: Segmentation failed for {label}: {e}")
            # Return rectangular segment as fallback
            x1, y1, x2, y2 = map(int, bbox)
            return [
                Point32(x=x1, y=y1, z=0), # Top-left
                Point32(x=x1, y=y2, z=0), # Bottom-left
                Point32(x=x2, y=y2, z=0), # Bottom-right
                Point32(x=x2, y=y1, z=0)  # Top-right
            ]

    def handle_service(self, req):
        # Example: Use the integer input to select an image file
        image_file = f"{self.image_path}image_{req.input}.jpg"
        try:
            import cv2
            cv_image = cv2.imread(image_file)
            if cv_image is None:
                return Florence2Response(output="Image not found")
            inputs = self.processor(images=cv_image, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=1024, num_beams=1, use_cache=False)
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return Florence2Response(output=result)
        except Exception as e:
            return Florence2Response(output=f"Error: {str(e)}")
        
    def general_service(self, req, prompt="<OD>"):
        start_time = time.time()
        image_file = f"{self.image_path}{req.input}.jpg"
        try:
            cv_image = cv2.imread(image_file)
            if cv_image is None:
                return Florence2Response(output="Image not found")
            
            # Apply letterboxing
            letterboxed_image, transform_params = self.letterbox_image(cv_image, target_size=768)
            letterbox_height, letterbox_width = letterboxed_image.shape[:2]
            
            inputs = self.processor(images=letterboxed_image, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)
            
            # Adjust generation parameters based on device
            max_tokens = 512 if self.device == "cpu" else 1024
            num_beams = 1  # Force greedy decoding to avoid cache issues
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                num_beams=num_beams,
                early_stopping=False,  # Disable early stopping for greedy decoding
                do_sample=False,       # Ensure deterministic output
                use_cache=False        # Disable cache to avoid cache layer issues
            )
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            elapsed = time.time() - start_time
            print(f"Service took {elapsed:.3f} seconds.")
            return Florence2Response(output=result)
        except Exception as e:
            return Florence2Response(output=f"Error: {str(e)}")

    def describe_scene(self, req):
        start_time = time.time()
        image_file = f"{self.image_path}{req.input}.jpg"
        try:
            cv_image = cv2.imread(image_file)
            if cv_image is None:
                return Florence2Response(output="Image not found")
            counts = self.extract_objects_and_counts(req.input)
            # Compose a prompt for Florence-2
            if counts:
            # Build a readable object list, e.g., "3 chair and 1 table"
                object_phrases = []
                for obj, count in counts.items():
                    object_phrases.append(f"{count} {obj}{'' if count == 1 else 's'}")
                if len(object_phrases) > 1:
                    object_str = ', '.join(object_phrases[:-1]) + ' and ' + object_phrases[-1]
                else:
                    object_str = object_phrases[0]
                prompt = f"<MORE_DETAILED_CAPTION>"
                #prompt = f"Describe the {object_str} in this image and explain their spatial relationships using prepositions such as 'on', 'under', 'next to', etc."
            else:
                prompt = (
                    #"Describe the scene in this image and list any relationships between objects. "
                    #"For example, mention if one object is on top of another, next to, or inside another."
                    #"Describe all objects in this image and explain their spatial relationships."
                    "<MORE_DETAILED_CAPTION>"
                )
            # Florence-2 expects the prompt as text input (if supported by your model)
            print(f"Using prompt: {prompt}")
            
            # Apply letterboxing
            letterboxed_image, transform_params = self.letterbox_image(cv_image, target_size=768)
            
            inputs = self.processor(images=letterboxed_image, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)
            
            # Adjust generation parameters based on device
            max_tokens = 512 if self.device == "cpu" else 1024
            num_beams = 1  # Force greedy decoding to avoid cache issues
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                num_beams=num_beams,
                early_stopping=False,  # Disable early stopping for greedy decoding
                do_sample=False,       # Ensure deterministic output
                use_cache=False        # Disable cache to avoid cache layer issues
            )
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            elapsed = time.time() - start_time
            print(f"Service took {elapsed:.3f} seconds.")
            return Florence2Response(output=result)
        except Exception as e:
            return Florence2Response(output=f"Error: {str(e)}")
    
    def extract_objects_and_counts(self, filename):
        # Remove the extension if present
        name = filename.split('.')[0]
        # Split off the index (last underscore part)
        parts = name.rsplit('_', 1)
        if len(parts) != 2:
            return {}
        objects_part = parts[0]
        object_list = objects_part.split('_')
        counts = {}
        for obj in object_list:
            counts[obj] = counts.get(obj, 0) + 1
        return counts
    
    def run_object_detection(self, image_path):
        start_time = time.time()
        try:
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                print("Image not found")
                return None
            # Example: Run Florence-2 object detection (replace with your actual detection logic)
            inputs = self.processor(images=cv_image, text="<OD>", return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            #print("Object Detection Result:", result)
        except Exception as e:
            print(f"Error during object detection: {e}")
        elapsed = time.time() - start_time
        print(f"Object detection took {elapsed:.3f} seconds.")

    def run_region_to_description(self, image_path, region=None):
        start_time = time.time()
        try:
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                print("Image not found")
                return None
            # Optionally crop to region if provided
            #if region:
            #    x1, y1, x2, y2 = region
            #    cv_image = cv_image[y1:y2, x1:x2]
            # Example: Run Florence-2 captioning
            prompt = "<REGION_TO_DESCRIPTION>"
            inputs = self.processor(images=cv_image, text=prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print("Region to Description Result:", result)
        except Exception as e:
            print(f"Error during region-to-description: {e}")
        elapsed = time.time() - start_time
        print(f"Region-to-description took {elapsed:.3f} seconds.")

    def run_detection_and_describe_regions(self, image_path):
        # Run object detection and get the result
        start_time = time.time()
        try:
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                print("Image not found")
                return
            # Run detection and parse result
            inputs = self.processor(images=cv_image, text="<OD>", return_tensors="pt")
            outputs = self.model.generate(**inputs)
            result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            # Assume result is a string that can be eval'd to a dict (or adapt this to your actual output)
            try:
                detection_dict = eval(result) if isinstance(result, str) else result
            except Exception as e:
                print(f"Could not parse detection result: {e}")
                return

            od_result = detection_dict.get('<OD>', {})
            bboxes = od_result.get('bboxes', [])
            labels = od_result.get('labels', [])

            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = map(int, bbox)
                region = cv_image[y1:y2, x1:x2]
                # Save region to a temporary file or pass directly if supported
                print(f"Running region-to-description for label '{label}' at bbox {bbox}")
                # Option 1: Pass region as image array (if your processor supports it)
                try:
                    prompt = f"<REGION_TO_DESCRIPTION> Describe the {label} in this region."
                    inputs = self.processor(images=region, text=prompt, return_tensors="pt")
                    outputs = self.model.generate(**inputs, max_new_tokens=128)
                    region_result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    print(f"Region description for '{label}': {region_result}")
                except Exception as e:
                    print(f"Error describing region for '{label}': {e}")

        except Exception as e:
            print(f"Error in detection and region description: {e}")
        elapsed = time.time() - start_time
        print(f"Detection and region-to-description took {elapsed:.3f} seconds.")
    
    def test_from_console(self):
        """Enhanced console test with option to save and quit"""
        print("Florence-2 Console Test Interface")
        print("Commands:")
        print("  <index> - Describe scene from image")
        print("  <index> <prompt> - Run custom prompt")
        print("  'save' - Save current collection and show stats")
        print("  'stats' - Show current collection statistics")
        print("  'q' - Quit and save")
        print()
        
        while True:
            try:
                user_input = input("Enter command: ")
                if user_input.lower() == 'q':
                    self.save_collection_and_exit()
                    break
                elif user_input.lower() == 'save':
                    self.save_collection_and_exit()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_collection_stats()
                    continue
                    
                parts = user_input.strip().split()
                if len(parts) == 1:
                    idx = parts[0]
                    class Req: pass
                    req = Req()
                    req.input = idx
                    resp = self.describe_scene(req)
                    print("Result:", resp.output)
                elif len(parts) == 2:
                    idx1, prompt = parts
                    class Req: pass
                    req1 = Req()
                    req1.input = idx1
                    self.general_service(req1, prompt)
                else:
                    print("Invalid command. Use 'q' to quit, 'save' to save, or provide image index.")
            except Exception as e:
                print(f"Error: {e}")
                
    def show_collection_stats(self):
        """Show current collection statistics"""
        if not self.detected_objects_collection:
            print("No objects detected yet.")
            return
            
        class_counts = self.create_class_count_dictionary()
        print(f"\nCurrent Detection Statistics:")
        print(f"Total objects detected: {len(self.detected_objects_collection)}")
        print(f"Unique classes: {len(class_counts)}")
        print("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
        print()

"""
if __name__ == '__main__':
    rospy.init_node('florence2_node')
    node = Florence2Node()
    rospy.loginfo("Florence2 service is ready.")
    rospy.spin()
"""
if __name__ == '__main__':
    if '--console' in sys.argv:
        node = Florence2Node()
        node.test_from_console()
    else:
        rospy.init_node('florence2_node')
        node = Florence2Node()
        rospy.loginfo("Florence2 service is ready.")
        
        # Register shutdown hook
        rospy.on_shutdown(node.save_collection_and_exit)
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            node.save_collection_and_exit()