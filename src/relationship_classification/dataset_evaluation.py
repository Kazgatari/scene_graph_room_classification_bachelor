#!/usr/bin/env python3
"""
Matterport3D Dataset Evaluation Script
Processes Matterport3D RGB and depth images with camera poses and intrinsics
to detect objects and calculate their 3D world positions.
"""

import rospy
import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from scene_graph.srv import VLMInference, VLMInferenceRequest
import struct


class Matterport3DProcessor:
    def __init__(self, matterport_path="/data/matterport", results_path="/data/results"):
        """Initialize the Matterport3D dataset processor"""
        rospy.init_node('matterport_dataset_evaluation', anonymous=True)
        
        self.matterport_path = Path(matterport_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize VLM service client
        self.vlm_service = None
        self.init_vlm_service()
        
        # Object ID counter
        self.object_id_counter = 0
        self.image_id_counter = 0
        
        rospy.loginfo(f"Matterport3D Processor initialized")
        rospy.loginfo(f"Dataset path: {self.matterport_path}")
        rospy.loginfo(f"Results path: {self.results_path}")
    
    def init_vlm_service(self):
        """Initialize VLM service client"""
        try:
            rospy.loginfo("Waiting for VLM service...")
            rospy.wait_for_service('vlm_inference', timeout=10.0)
            self.vlm_service = rospy.ServiceProxy('vlm_inference', VLMInference)
            rospy.loginfo("VLM service connected successfully")
        except rospy.ROSException:
            rospy.logerr("VLM service not available")
            self.vlm_service = None
        except Exception as e:
            rospy.logerr(f"Error initializing VLM service: {e}")
            self.vlm_service = None
    
    def load_camera_intrinsics(self, intrinsics_file):
        """Load camera intrinsics from file
        
        Format: width height fx fy cx cy k1 k2 p1 p2 k3
        
        Returns:
            dict: Camera intrinsics parameters
        """
        try:
            with open(intrinsics_file, 'r') as f:
                params = f.read().strip().split()
                params = [float(p) for p in params]
            
            intrinsics = {
                'width': int(params[0]),
                'height': int(params[1]),
                'fx': params[2],
                'fy': params[3],
                'cx': params[4],
                'cy': params[5],
                'k1': params[6],
                'k2': params[7],
                'p1': params[8],
                'p2': params[9],
                'k3': params[10]
            }
            
            return intrinsics
            
        except Exception as e:
            rospy.logerr(f"Error loading camera intrinsics from {intrinsics_file}: {e}")
            return None
    
    def load_camera_pose(self, pose_file):
        """Load camera pose (4x4 transformation matrix) from file
        
        Transforms column vectors from camera to global coordinates.
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        try:
            pose_matrix = []
            with open(pose_file, 'r') as f:
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    pose_matrix.append(values)
            
            return np.array(pose_matrix)
            
        except Exception as e:
            rospy.logerr(f"Error loading camera pose from {pose_file}: {e}")
            return None
    
    def load_depth_image(self, depth_file):
        """Load depth image (16-bit PNG)
        
        Depth values are in 0.25mm per value (divide by 4000 to get meters).
        Zero values denote 'no reading'.
        
        Returns:
            np.ndarray: Depth image in meters
        """
        try:
            # Load 16-bit depth image
            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
            
            if depth_img is None:
                rospy.logerr(f"Failed to load depth image: {depth_file}")
                return None
            
            # Convert to meters (divide by 4000)
            depth_meters = depth_img.astype(np.float32) / 4000.0
            
            # Set zero values (no reading) to NaN
            depth_meters[depth_meters == 0] = np.nan
            
            return depth_meters
            
        except Exception as e:
            rospy.logerr(f"Error loading depth image from {depth_file}: {e}")
            return None
    
    def load_rgb_image(self, rgb_file):
        """Load RGB image (tone-mapped color image)
        
        Returns:
            np.ndarray: RGB image
        """
        try:
            rgb_img = cv2.imread(str(rgb_file))
            
            if rgb_img is None:
                rospy.logerr(f"Failed to load RGB image: {rgb_file}")
                return None
            
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            return rgb_img
            
        except Exception as e:
            rospy.logerr(f"Error loading RGB image from {rgb_file}: {e}")
            return None
    
    def rgb_to_ros_image(self, rgb_img):
        """Convert OpenCV RGB image to ROS Image message"""
        try:
            return self.bridge.cv2_to_imgmsg(rgb_img, encoding="rgb8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB to ROS Image: {e}")
            return None
    
    def pixel_to_camera_coords(self, pixel_x, pixel_y, depth, intrinsics):
        """Convert pixel coordinates + depth to 3D camera coordinates
        
        Args:
            pixel_x: Pixel x coordinate
            pixel_y: Pixel y coordinate
            depth: Depth value in meters (z-direction from camera center)
            intrinsics: Camera intrinsics dict
        
        Returns:
            np.ndarray: 3D point in camera frame [x, y, z]
        """
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['cx']
        cy = intrinsics['cy']
        
        # Convert pixel to camera coordinates
        # Note: depth is already z-direction distance from camera center
        camera_x = (pixel_x - cx) * depth / fx
        camera_y = (pixel_y - cy) * depth / fy
        camera_z = depth
        
        return np.array([camera_x, camera_y, camera_z])
    
    def camera_to_world_coords(self, camera_point, pose_matrix):
        """Transform point from camera frame to world frame
        
        Args:
            camera_point: 3D point in camera frame [x, y, z]
            pose_matrix: 4x4 camera pose transformation matrix
        
        Returns:
            np.ndarray: 3D point in world frame [x, y, z]
        """
        # Convert to homogeneous coordinates
        camera_point_h = np.append(camera_point, 1.0)
        
        # Transform to world coordinates
        world_point_h = pose_matrix @ camera_point_h
        
        # Convert back to 3D coordinates
        return world_point_h[:3]
    
    def sample_depth_in_bbox(self, bbox, depth_img, rgb_img_shape):
        """Sample depth values within bounding box
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates (based on RGB image)
            depth_img: Depth image in meters
            rgb_img_shape: Shape of RGB image (height, width) for coordinate scaling
        
        Returns:
            list: Valid depth values in the bounding box
        """
        x1, y1, x2, y2 = bbox
        
        # If depth and RGB have different dimensions, scale bbox coordinates
        rgb_height, rgb_width = rgb_img_shape[:2]
        depth_height, depth_width = depth_img.shape[:2]
        
        if (depth_height != rgb_height) or (depth_width != rgb_width):
            # Scale bbox from RGB coordinates to depth coordinates
            scale_x = depth_width / rgb_width
            scale_y = depth_height / rgb_height
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
        
        # Convert to integers and ensure within depth image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(depth_img.shape[1] - 1, int(x2))
        y2 = min(depth_img.shape[0] - 1, int(y2))
        
        # Extract depth values in bounding box
        depth_roi = depth_img[y1:y2, x1:x2]
        
        # Get valid (non-NaN) depth values
        valid_depths = depth_roi[~np.isnan(depth_roi)]
        
        if len(valid_depths) > 0:
            return valid_depths.tolist()
        else:
            # Default to 2m if no valid depth (common at edges/reflective surfaces)
            rospy.logdebug(f"No valid depth in bbox [{x1},{y1},{x2},{y2}], using default 2.0m")
            return [2.0]
    
    def estimate_object_size_from_bbox(self, bbox, depth, intrinsics):
        """Estimate 3D object size from 2D bounding box and depth
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            depth: Median depth of the object
            intrinsics: Camera intrinsics
        
        Returns:
            np.ndarray: Object size [width, height, depth]
        """
        x1, y1, x2, y2 = bbox
        
        # Calculate 2D size in pixels
        width_pixels = abs(x2 - x1)
        height_pixels = abs(y2 - y1)
        
        # Ensure minimum pixel size
        width_pixels = max(width_pixels, 5.0)
        height_pixels = max(height_pixels, 5.0)
        
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        
        # Convert pixel dimensions to real-world dimensions
        width_real = (width_pixels * depth) / fx
        height_real = (height_pixels * depth) / fy
        
        # Estimate depth dimension using heuristics
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
    
    def create_bounding_box(self, centroid, size):
        """Create 3D bounding box min and max corners
        
        Args:
            centroid: Object center position [x, y, z]
            size: Object size [width, height, depth]
        
        Returns:
            dict: Bounding box with min_corner and max_corner
        """
        half_size = size / 2.0
        
        min_corner = {
            'x': float(centroid[0] - half_size[0]),
            'y': float(centroid[1] - half_size[1]),
            'z': float(centroid[2] - half_size[2])
        }
        
        max_corner = {
            'x': float(centroid[0] + half_size[0]),
            'y': float(centroid[1] + half_size[1]),
            'z': float(centroid[2] + half_size[2])
        }
        
        return {'min_corner': min_corner, 'max_corner': max_corner}
    
    def call_vlm_service(self, rgb_image):
        """Call VLM service for object detection
        
        Args:
            rgb_image: RGB image as numpy array
        
        Returns:
            VLMInferenceResponse or None
        """
        if self.vlm_service is None:
            rospy.logerr("VLM service not available")
            return None
        
        try:
            # Convert to ROS Image message
            ros_image = self.rgb_to_ros_image(rgb_image)
            if ros_image is None:
                return None
            
            # Create service request
            request = VLMInferenceRequest()
            request.image = ros_image
            
            # Call service
            rospy.loginfo("Calling VLM service for object detection...")
            response = self.vlm_service(request)
            
            if response.success:
                rospy.loginfo(f"VLM detected {len(response.scene_graphs)} objects")
                return response
            else:
                rospy.logerr(f"VLM inference failed: {response.error_message}")
                return None
                
        except Exception as e:
            rospy.logerr(f"Error calling VLM service: {e}")
            return None
    
    def process_image(self, rgb_file, depth_file, pose_file, intrinsics_file):
        """Process a single image with its corresponding data
        
        Args:
            rgb_file: Path to RGB image
            depth_file: Path to depth image
            pose_file: Path to camera pose file
            intrinsics_file: Path to camera intrinsics file
        
        Returns:
            list: Detected objects in JSON format
        """
        rospy.loginfo(f"Processing image {self.image_id_counter}: {rgb_file.name}")
        self.image_id_counter += 1

        # Load all data
        rgb_img = self.load_rgb_image(rgb_file)
        depth_img = self.load_depth_image(depth_file)
        pose_matrix = self.load_camera_pose(pose_file)
        intrinsics = self.load_camera_intrinsics(intrinsics_file)
        
        if rgb_img is None or depth_img is None or pose_matrix is None or intrinsics is None:
            rospy.logerr(f"Failed to load data for {rgb_file.name}")
            return []
        
        # Call VLM service for object detection
        vlm_response = self.call_vlm_service(rgb_img)
        
        if vlm_response is None or not vlm_response.success:
            rospy.logwarn(f"No detections for {rgb_file.name}")
            return []
        
        detected_objects = []
        
        # Process each detected object
        for scene_graph in vlm_response.scene_graphs:
            try:
                # Get bounding box
                if len(scene_graph.bbox_2d) != 4:
                    rospy.logwarn(f"Invalid bbox for object {scene_graph.main_object.name}")
                    continue
                
                bbox_raw = scene_graph.bbox_2d
                img_height, img_width = rgb_img.shape[:2]
                
                # Detect if bbox values are normalized [0-1] or percentile [0-999]
                max_bbox_val = max(bbox_raw)
                
                if max_bbox_val <= 1.0:
                    # Normalized format [0-1]: directly multiply by image dimensions
                    rospy.logdebug(f"  Detected normalized bbox format [0-1]: {bbox_raw}")
                    x1 = bbox_raw[0] * img_width
                    y1 = bbox_raw[1] * img_height
                    x2 = bbox_raw[2] * img_width
                    y2 = bbox_raw[3] * img_height
                else:
                    # Percentile format [0-999]: divide by 999 then multiply by dimensions
                    rospy.logdebug(f"  Detected percentile bbox format [0-999]: {bbox_raw}")
                    x1 = (bbox_raw[0] / 999.0) * img_width
                    y1 = (bbox_raw[1] / 999.0) * img_height
                    x2 = (bbox_raw[2] / 999.0) * img_width
                    y2 = (bbox_raw[3] / 999.0) * img_height
                
                # Ensure proper ordering (x1 < x2, y1 < y2)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Validate bbox has minimum size (at least 5 pixels in each dimension)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                min_bbox_size = 5
                
                if bbox_width < min_bbox_size or bbox_height < min_bbox_size:
                    rospy.logdebug(f"Skipping {scene_graph.main_object.name}: bbox too small or out of bounds (raw={bbox_raw}, size={bbox_width:.0f}x{bbox_height:.0f})")
                    continue
                
                rospy.logdebug(f"  {scene_graph.main_object.name}: raw_bbox={bbox_raw} -> pixels=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
                
                pixel_bbox = [x1, y1, x2, y2]
                
                # Get center of bounding box
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                # Sample depth in bounding box (pass RGB image shape for coordinate scaling)
                depth_samples = self.sample_depth_in_bbox(pixel_bbox, depth_img, rgb_img.shape)
                median_depth = float(np.median(depth_samples))
                
                # Convert center pixel to camera coordinates
                camera_point = self.pixel_to_camera_coords(
                    center_x, center_y, median_depth, intrinsics
                )
                
                # Transform to world coordinates
                world_point = self.camera_to_world_coords(camera_point, pose_matrix)
                
                # Estimate object size
                object_size = self.estimate_object_size_from_bbox(
                    pixel_bbox, median_depth, intrinsics
                )
                
                # Create bounding box
                bounding_box = self.create_bounding_box(world_point, object_size)
                
                # Get object attributes
                object_name = scene_graph.main_object.name
                color = scene_graph.main_object.attributes.color if scene_graph.main_object.attributes else ""
                style = scene_graph.main_object.attributes.style if scene_graph.main_object.attributes else ""
                
                # Extract relations
                relations = []
                for env_obj in scene_graph.environment:
                    if env_obj.name and env_obj.attributes.style:  # style contains relation type
                        relations.append({
                            'type': env_obj.attributes.style,
                            'target': env_obj.name
                        })
                
                # Create object JSON
                obj_json = {
                    'id': f"obj_{self.object_id_counter}",
                    'semantic_label': object_name,
                    'centroid': {
                        'x': float(world_point[0]),
                        'y': float(world_point[1]),
                        'z': float(world_point[2])
                    },
                    'bounding_box': bounding_box,
                    'attributes': {
                        'color': color,
                        'relations': relations
                    },
                    'style': style,
                    'rgb_file': str(rgb_file.name)
                }
                
                detected_objects.append(obj_json)
                self.object_id_counter += 1
                
                rospy.loginfo(f"  Detected: {object_name} at ({world_point[0]:.2f}, {world_point[1]:.2f}, {world_point[2]:.2f})")
                
            except Exception as e:
                rospy.logerr(f"Error processing detection: {e}")
                continue
        
        return detected_objects
    
    def process_panorama(self, panorama_uuid, color_path, depth_path, pose_path, intrinsics_path):
        """Process all images for a single panorama
        
        Args:
            panorama_uuid: Unique panorama identifier
            color_path: Path to color images directory
            depth_path: Path to depth images directory
            pose_path: Path to camera poses directory
            intrinsics_path: Path to camera intrinsics directory
        
        Returns:
            list: All detected objects from this panorama
        """
        rospy.loginfo(f"Processing panorama: {panorama_uuid}")
        
        all_objects = []
        
        # Process all 18 images (3 cameras x 6 yaw positions)
        for camera_idx in range(3):
            for yaw_idx in range(6):
                # Construct filenames
                rgb_file = color_path / f"{panorama_uuid}_i{camera_idx}_{yaw_idx}.jpg"
                depth_file = depth_path / f"{panorama_uuid}_d{camera_idx}_{yaw_idx}.png"
                pose_file = pose_path / f"{panorama_uuid}_pose_{camera_idx}_{yaw_idx}.txt"
                intrinsics_file = intrinsics_path / f"{panorama_uuid}_intrinsics_{camera_idx}.txt"
                
                # Check if files exist
                if not all([rgb_file.exists(), depth_file.exists(), 
                           pose_file.exists(), intrinsics_file.exists()]):
                    rospy.logwarn(f"Missing files for {panorama_uuid}_i{camera_idx}_{yaw_idx}")
                    continue
                
                # Process this image
                objects = self.process_image(rgb_file, depth_file, pose_file, intrinsics_file)
                all_objects.extend(objects)
        
        return all_objects
    
    def save_results(self, objects, output_file):
        """Save detected objects to JSON file
        
        Args:
            objects: List of detected object dictionaries
            output_file: Path to output JSON file
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(objects, f, indent=2)
            
            rospy.loginfo(f"Saved {len(objects)} objects to {output_file}")
            
        except Exception as e:
            rospy.logerr(f"Error saving results to {output_file}: {e}")
    
    def process_building(self):
        """Process all panoramas in the Matterport building
        
        Data is organized in separate directories for each data type.
        """
        rospy.loginfo(f"Processing Matterport building from: {self.matterport_path}")
        
        # Define paths to data directories
        color_path = self.matterport_path / "matterport_color_images"
        depth_path = self.matterport_path / "matterport_depth_images"
        pose_path = self.matterport_path / "matterport_camera_poses"
        intrinsics_path = self.matterport_path / "matterport_camera_intrinsics"
        
        # Check if directories exist
        if not all([color_path.exists(), depth_path.exists(), 
                   pose_path.exists(), intrinsics_path.exists()]):
            rospy.logerr("One or more required directories are missing:")
            rospy.logerr(f"  Color images: {color_path.exists()}")
            rospy.logerr(f"  Depth images: {depth_path.exists()}")
            rospy.logerr(f"  Camera poses: {pose_path.exists()}")
            rospy.logerr(f"  Camera intrinsics: {intrinsics_path.exists()}")
            return
        
        # Find all unique panorama UUIDs by looking at RGB images
        rgb_files = list(color_path.glob("*_i0_0.jpg"))
        panorama_uuids = [f.name.split('_i0_0.jpg')[0] for f in rgb_files]
        
        rospy.loginfo(f"Found {len(panorama_uuids)} panoramas to process")
        
        all_objects = []
        
        # Process each panorama
        for panorama_uuid in panorama_uuids:
            objects = self.process_panorama(
                panorama_uuid, 
                color_path, 
                depth_path, 
                pose_path, 
                intrinsics_path
            )
            all_objects.extend(objects)
        
        # Save results
        output_file = self.results_path / "matterport_objects.json"
        self.save_results(all_objects, output_file)
        
        rospy.loginfo(f"Finished processing: {len(all_objects)} total objects detected")
    
    def run(self):
        """Main processing loop - process the Matterport building"""
        rospy.loginfo("Starting Matterport3D dataset processing...")
        
        if not self.matterport_path.exists():
            rospy.logerr(f"Matterport path does not exist: {self.matterport_path}")
            return
        
        try:
            self.process_building()
        except Exception as e:
            rospy.logerr(f"Error processing Matterport building: {e}")
            import traceback
            traceback.print_exc()
        
        rospy.loginfo("Dataset processing complete!")


def main():
    try:
        # Initialize processor
        processor = Matterport3DProcessor(
            matterport_path="/data/matterport",
            results_path="/data/results"
        )
        
        # Run processing
        processor.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Matterport3D processor interrupted")
    except Exception as e:
        rospy.logerr(f"Error in Matterport3D processor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
