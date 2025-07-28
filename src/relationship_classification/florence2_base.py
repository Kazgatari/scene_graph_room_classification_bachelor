#!/usr/bin/env python3
from cProfile import label

import numpy as np
import rospy
from torch import device
import torch
#from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoProcessor, AutoModelForCausalLM
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import time
from scene_graph.srv import Florence2, Florence2Response
import sys
from scene_graph.msg import DetectedObjects, DetectedObject
from geometry_msgs.msg import Point32
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
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

        self.device = "cuda" # Change to "cuda" if gpu or : "cuda" if torch.cuda.is_available()  // or "cpu" for CPU
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        
        self.model = self.model.to(self.device, dtype=self.torch_dtype) # if cuda
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
            self.torch_dtype = torch.float32
        
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
        self.detected_objects_pub = rospy.Publisher('/scene_graph/detected_objects', DetectedObjects, queue_size=10)

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
            pickle_file = os.path.join(self.output_folder, f"detected_objects_{timestamp}.pkl")
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.detected_objects_collection, f)
            
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
                f.write(f"\nData saved to:\n")
                f.write(f"  - JSON: {json_file}\n")
                f.write(f"  - Pickle: {pickle_file}\n")
                f.write(f"  - Counts: {counts_file}\n")
            
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

        results = self.run_general(task_prompt="<DENSE_REGION_CAPTION>", cv_image=cv_image)
        if results is None:
            rospy.logwarn("No detection results received")
            return
        
        bboxes = results['bboxes']
        labels = results['labels']
        
        # Store detection timestamp
        detection_timestamp = rospy.Time.now().to_sec()
        
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            detected_object = DetectedObject()
            detected_object.class_name = String(data=label)
            detected_object.image_index = Int32(data=0)
            detected_object.bounding_box = [
                Point32(x=x1, y=y1, z=0),
                Point32(x=x2, y=y2, z=0)
            ]
            #segment_points = self.get_segmentation_mask(cv_image, bbox, label)
            detected_object.segment = []#segment_points
            detected_object.description = String(data=label)

            detected_objects.append(detected_object)
            
            # Add to collection for saving later
            detection_data = {
                'timestamp': detection_timestamp,
                'class_name': label,
                'bounding_box': {
                    'x1': float(x1),
                    'y1': float(y1), 
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'image_dimensions': {
                    'width': cv_image.shape[1],
                    'height': cv_image.shape[0]
                }
            }
            self.detected_objects_collection.append(detection_data)

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
        #cv_image = image_msg #self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        height, width = cv_image.shape[:2]
        print(f"Image dimensions: {width}x{height}")

        inputs = self.processor(text=prompt, images=cv_image, return_tensors="pt").to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          max_new_tokens=1024,
          num_beams=3,
          early_stopping=True
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        #height, width = cv_image.shape[:2]

        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(width, height))

        if task_prompt == "<OD>":
            od_result = parsed_answer['<OD>']
            #return od_result

        # Debug: Print bounding boxes
            for i, (bbox, label) in enumerate(zip(od_result['bboxes'], od_result['labels'])):
                x1, y1, x2, y2 = bbox
                print(f"Object {i}: {label} at ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")

                # Check if coordinates are within image bounds
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    print(f"  WARNING: Coordinates outside image bounds!")

            return od_result
        
        elif task_prompt == "<DENSE_REGION_CAPTION>":
            od_result = parsed_answer['<DENSE_REGION_CAPTION>']
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
                num_beams=1,         # Reduced from 3 for faster processing
                do_sample=False,     # Deterministic output
                early_stopping=False
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
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
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
            inputs = self.processor(images=cv_image, text=prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
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
            inputs = self.processor(images=cv_image, text=prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
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