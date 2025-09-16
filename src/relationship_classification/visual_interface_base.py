#!/usr/bin/env python3
import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry
import numpy as np
import struct
from geometry_msgs.msg import Point32
from std_msgs.msg import String, Int32, Header
from typing import List
from scene_graph.srv import VLMInference, VLMInferenceRequest
from scene_graph.msg import GraphObjects, GraphObject


class VisualInterfaceBase:
    def __init__(self):
        """Synchronized input subscribers"""
        self.image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/points', PointCloud2)
        self.odom_sub = message_filters.Subscriber('/odom', Odometry)

        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.odom_sub], queue_size=100, slop=0.1)
        ts.registerCallback(self.synchronized_callback)

        """ Add camera info subscriber for 3D coordinate calculation """
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        self.camera_info = None
        # Default camera intrinsics (will be updated from camera_info)
        self.fx = 615.0  # focal length x
        self.fy = 615.0  # focal length y
        self.cx = 320.0  # principal point x
        self.cy = 240.0  # principal point y

        # Publisher for GraphObjects
        self.graph_objects_pub = rospy.Publisher('/scene_graph/seen_graph_objects', GraphObjects, queue_size=10)
        
        # VLM service client
        self.vlm_service = None
        self.init_vlm_service()
        
        # Object ID counter for unique identification
        self.object_id_counter = 0

    def init_vlm_service(self):
        """Initialize VLM service client with robust error handling"""
        try:
            rospy.wait_for_service('vlm_inference', timeout=2.0)
            self.vlm_service = rospy.ServiceProxy('vlm_inference', VLMInference)
            rospy.loginfo("VLM service connected successfully")
        except rospy.ROSException:
            rospy.logwarn("VLM service not available, will retry on each callback")
            self.vlm_service = None
        except Exception as e:
            rospy.logerr(f"Error initializing VLM service: {e}")
            self.vlm_service = None

    
    def camera_info_callback(self, msg):
        """Store camera intrinsics for 3D coordinate calculation"""
        self.camera_info = msg
        self.fx = msg.K[0]  
        self.fy = msg.K[4]  
        self.cx = msg.K[2]  
        self.cy = msg.K[5] 

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
        
        # Convert to 3D camera coordinates
        camera_point = self.pixel_to_camera_coords(center_x, center_y, estimated_depth)
        
        # Transform to world coordinates using odometry
        world_point = self.transform_to_world_coords(camera_point, odom_msg)
        
        return world_point

    def create_3d_bounding_box(self, position: np.ndarray, size: np.ndarray) -> List[Point32]:
        """Create 3D bounding box with min and max corners for graph management node"""
        # Sanitize size to ensure valid dimensions
        size = self.sanitize_object_size(size)
        
        # Calculate half extents
        half_x, half_y, half_z = size / 2.0
        
        # Create only min and max corners (2 points) as expected by graph management node
        min_corner = Point32(
            x=float(position[0] - half_x), 
            y=float(position[1] - half_y), 
            z=float(position[2] - half_z)
        )
        max_corner = Point32(
            x=float(position[0] + half_x), 
            y=float(position[1] + half_y), 
            z=float(position[2] + half_z)
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

    def estimate_object_size_from_bbox(self, bbox, depth):
        """Estimate 3D object size from 2D bounding box and depth with improved heuristics"""
        x1, y1, x2, y2 = bbox
        
        # Calculate 2D size in pixels
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        
        # Ensure minimum pixel size to avoid division issues
        width_pixels = max(width_pixels, 5.0)
        height_pixels = max(height_pixels, 5.0)
        
        # Convert pixel dimensions to real-world dimensions using depth
        # Real world size = (pixel size * depth) / focal length
        width_real = (width_pixels * depth) / self.fx
        height_real = (height_pixels * depth) / self.fy
        
        # Estimate depth dimension using object-specific heuristics
        # For most objects, assume roughly cubic proportions but with some variation
        aspect_ratio = width_pixels / height_pixels
        
        if aspect_ratio > 2.0:  # Wide objects (tables, shelves)
            depth_real = min(width_real, height_real) * 0.5
        elif aspect_ratio < 0.5:  # Tall objects (bottles, lamps)
            depth_real = width_real * 0.8
        else:  # Roughly square objects
            depth_real = min(width_real, height_real) * 0.7
        
        # Ensure reasonable bounds (5cm to 2m)
        width_real = np.clip(width_real, 0.05, 2.0)
        height_real = np.clip(height_real, 0.05, 2.0)
        depth_real = np.clip(depth_real, 0.05, 2.0)
        
        return np.array([width_real, height_real, depth_real])

    def create_graph_object_from_detection(self, detection_data, world_position, object_size, object_id):
        """Create a GraphObject from detection data"""
        graph_obj = GraphObject()
        
        # Set object name
        graph_obj.name = String()
        graph_obj.name.data = detection_data.get('label', 'unknown')
        
        # Set object ID
        graph_obj.object_id = Int32()
        graph_obj.object_id.data = object_id
        
        # Create 3D bounding box
        graph_obj.bounding_box = self.create_3d_bounding_box(world_position, object_size)
        
        return graph_obj

    def process_vlm_detections(self, image_msg, depth_msg, odom_msg):
        """Process VLM detections and return GraphObjects"""
        try:
            # Initialize VLM service if not available
            if self.vlm_service is None:
                self.init_vlm_service()
                if self.vlm_service is None:
                    rospy.logwarn_throttle(5.0, "VLM service still not available")
                    return []

            # Call VLM service
            request = VLMInferenceRequest()
            request.image = image_msg
            
            response = self.vlm_service(request)
            
            if not response.success:
                rospy.logwarn(f"VLM inference failed: {response.error_message}")
                return []
            
            graph_objects = []
            
            # Process each detected object
            for scene_graph in response.scene_graphs:
                # Get 2D bounding box directly from the ObjectSceneGraph
                if len(scene_graph.bbox_2d) == 4:
                    try:
                        # bbox_2d contains [x1, y1, x2, y2] in percentile values (0-999)
                        bbox_percentile = scene_graph.bbox_2d
                        
                        # Convert from percentile (0-999) to pixel coordinates
                        img_height = image_msg.height
                        img_width = image_msg.width
                        
                        x1 = (bbox_percentile[0] / 999.0) * img_width
                        y1 = (bbox_percentile[1] / 999.0) * img_height
                        x2 = (bbox_percentile[2] / 999.0) * img_width
                        y2 = (bbox_percentile[3] / 999.0) * img_height
                        
                        pixel_bbox = [x1, y1, x2, y2]
                        
                        # Calculate 3D world position
                        world_position = self.calculate_3d_world_position(pixel_bbox, depth_msg, odom_msg)
                        
                        # Calculate depth for size estimation
                        depth_samples = self.sample_depth_in_bbox(pixel_bbox, depth_msg)
                        estimated_depth = np.median(depth_samples)
                        
                        # Estimate 3D object size
                        object_size = self.estimate_object_size_from_bbox(pixel_bbox, estimated_depth)
                        
                        # Validate detection data
                        if not self.validate_detection_data(pixel_bbox, world_position, object_size):
                            rospy.logwarn(f"Invalid detection data for object {scene_graph.main_object.name}, skipping")
                            continue
                        
                        # Create GraphObject
                        self.object_id_counter += 1
                        graph_obj = self.create_graph_object_from_detection(
                            {'label': scene_graph.main_object.name},
                            world_position,
                            object_size,
                            self.object_id_counter
                        )
                        
                        # Log detection info for debugging
                        self.log_detection_info(graph_obj, world_position, object_size)
                        
                        graph_objects.append(graph_obj)
                
                    except (ValueError, IndexError) as e:
                        rospy.logwarn(f"Error processing 2D bounding box {scene_graph.bbox_2d}: {e}")
                        continue
                else:
                    rospy.logwarn(f"Invalid 2D bounding box format for object {scene_graph.main_object.name}: {scene_graph.bbox_2d}")
                    continue
            
            return graph_objects
            
        except Exception as e:
            rospy.logerr(f"Error processing VLM detections: {e}")
            return []

    def validate_detection_data(self, bbox, world_position, object_size):
        """Validate detection data for reasonable values"""
        # Check bounding box
        if len(bbox) != 4 or any(coord < 0 for coord in bbox):
            return False
        
        # Check world position (should be finite)
        if not np.all(np.isfinite(world_position)):
            return False
        
        # Check object size (should be positive and reasonable)
        if not np.all(object_size > 0) or np.any(object_size > 5.0):  # Max 5m in any dimension
            return False
        
        return True

    def log_detection_info(self, graph_obj, world_position, object_size):
        """Log detailed information about detected objects"""
        name = graph_obj.name.data
        obj_id = graph_obj.object_id.data
        rospy.logdebug(f"Detected object '{name}' (ID: {obj_id}):")
        rospy.logdebug(f"  Position: [{world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f}]")
        rospy.logdebug(f"  Size: [{object_size[0]:.2f}, {object_size[1]:.2f}, {object_size[2]:.2f}]")
        rospy.logdebug(f"  Bounding box corners: {len(graph_obj.bounding_box)} points")

    def synchronized_callback(self, image_msg, depth_msg, odom_msg):
        """Process synchronized sensor data and publish detected objects"""
        try:
            # Process VLM detections
            graph_objects = self.process_vlm_detections(image_msg, depth_msg, odom_msg)
            
            if graph_objects:
                # Create GraphObjects message
                graph_objects_msg = GraphObjects()
                graph_objects_msg.header = Header()
                graph_objects_msg.header.stamp = rospy.Time.now()
                # graph_objects_msg.header.frame_id = "world"  # or appropriate frame
                graph_objects_msg.objects = graph_objects
                
                # Publish GraphObjects
                self.graph_objects_pub.publish(graph_objects_msg)
                
                rospy.loginfo(f"Published {len(graph_objects)} detected objects")
            else:
                rospy.logdebug("No objects detected in current frame")
                
        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")

if __name__ == '__main__':
    rospy.init_node('visual_interface_node', anonymous=True)
    visual_interface = VisualInterfaceBase()
    rospy.loginfo("Visual Interface Node started, waiting for synchronized messages...")
    rospy.spin()