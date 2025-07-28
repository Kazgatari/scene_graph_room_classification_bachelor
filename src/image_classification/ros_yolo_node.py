#!/usr/bin/env python3

from ultralytics import YOLO
import rospy
import torch
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from scene_graph.msg import DetectedObjects, DetectedObject
from geometry_msgs.msg import Point32
from std_msgs.msg import String
from nav_msgs.msg import Odometry
#import collections # Igor
from std_msgs.msg import Int32
from collections import deque
from collections import Counter
import os

#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2

class YOLOv9SegNode:
    
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolov9_seg_node', anonymous=True)

        # Load YOLOv9 model
        self.model = YOLO('yolov9e-seg.pt')
        # self.model.to('cuda')
        # self.model.eval()
        self.n = 0
        self.in_cb = False
        
        self.rgb_image = Image()
        self.depth_msg = PointCloud2()
        self.odom_msg = Odometry()

        # Create a CvBridge object for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Subscribe to the camera image topic
        self.image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/points', PointCloud2)
        self.odom_sub = message_filters.Subscriber('/odom', Odometry)
        
        # Approximate Time Synchronizer allows slight time differences between topics
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.odom_sub], queue_size=100, slop=0.1)
        ts.registerCallback(self.synchronized_callback)

        # Publisher for the segmented image
        self.image_pub = rospy.Publisher('/scene_graph/color/image_raw', Image, queue_size=1)
        self.segmented_image_pub = rospy.Publisher('/camera/color/segmented_image', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/scene_graph/depth/points', PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher('/scene_graph/odom', Odometry, queue_size=1)
        
        self.detected_objects_pub = rospy.Publisher('/scene_graph/detected_objects', DetectedObjects, queue_size=10)
        
        #self.object_info = collections.defaultdict(list) # Igor: Store images for each detected object class
        # Images TODO: relative path
        os.makedirs('/root/catkin_ws/src/scene_graph_room_classification/images/detected_objects', exist_ok=True)
        self.image_index = 0
        self.image_path = '/root/catkin_ws/src/scene_graph_room_classification/images/detected_objects/'
        self.last_class_name = ''
        self.last_save_time = 0
        self.same_class_count = 0

        self.image_queue_size = 20
        self.image_queue_saved_size = 10
        self.image_queue = deque(maxlen=self.image_queue_size)
        self.image_queue_saved = deque(maxlen=self.image_queue_saved_size)
        self.queue_saved_empty = True
        self.comparison_time_range = 8.0 # seconds for image comparison with the last saved image

        # Performance optimization (Odom only if environment is static)
        self.pause_for_seconds = 0.1
        self.last_odom_position = None
        self.last_odom_orientation = None
        self.position_threshold = 0.05  # meters
        self.orientation_threshold = 0.05  # radians

    #Igor:Check for Odom position and orientation changes
    def has_significant_odom_change(self, current_odom):
        # Compare position
        pos = current_odom.pose.pose.position
        if self.last_odom_position is not None:
            dx = pos.x - self.last_odom_position.x
            dy = pos.y - self.last_odom_position.y
            dz = pos.z - self.last_odom_position.z
            position_change = (dx**2 + dy**2 + dz**2) ** 0.5
        else:
            position_change = float('inf')

        # Compare orientation (quaternion difference)
        ori = current_odom.pose.pose.orientation
        if self.last_odom_orientation is not None:
            # Compute the angle between quaternions
            import numpy as np
            q1 = np.array([ori.x, ori.y, ori.z, ori.w])
            q0 = np.array([self.last_odom_orientation.x, self.last_odom_orientation.y, self.last_odom_orientation.z, self.last_odom_orientation.w])
            dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
            angle = 2 * np.arccos(abs(dot))
        else:
            angle = float('inf')

        return position_change > self.position_threshold or angle > self.orientation_threshold

    def synchronized_callback(self, ros_image, depth_msg, odom_msg):
        # Convert ROS Image message to OpenCV format
        
        # depth_msg = self.depth_msg
        # odom_msg = self.odom_msg
        
        # self.n += 1
        
        # if not self.n % 10 == 0:
        #     return
        
        
        print('in synchronized callback')
        
        start_time = time.time()

        # Igor: Lessen amount of images processed TODO: Try image comparison instead // Check Synchronization
        """
        if ((start_time - self.last_save_time) < self.pause_for_seconds): #Time between processing images
            return
        elif not self.has_significant_odom_change(odom_msg): # Check for Odom position and orientation changes (Only if environment is static))
            return
        """


        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        self.rgb_image = ros_image

        # Perform segmentation using YOLOv9 model
        results = self.model(cv_image)
        
        # Extract segmentation masks and apply them to the original image
        masks = results[0].masks.data if results[0].masks is not None else None  # Assuming the output format includes masks in xyn format
        if masks is None:
            rospy.logwarn("No masks found in the image.")
        

        detected_objects = []
        indices = []
        #igor:
        object_detected = False
        image_path_combination = ''
        object_array_for_queue = []
        object_class_names_array = []
        
                        
        for result in results:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = box.conf.item()
                

                if confidence > 0.6:
                    x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()
                    print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                    
                    #Igor
                    if class_name in self.last_class_name:
                        image_index = self.image_index
                    else:
                        image_index = self.image_index + 1
                    
                    detected_object = DetectedObject(String(str(class_name)), [Point32(x1, y1, 0.0), Point32(x2, y2, 0.0)], [], Int32(image_index))
                    object_array_for_queue.append(detected_object)
                    object_class_names_array.append(class_name)
                    #self.image_queue.append(start_time, cv_image, DetectedObject(String(str(class_name)), [Point32(x1, y1, 0.0), Point32(x2, y2, 0.0)], [], image_index))
                    ###

                    detected_objects.append(DetectedObject(String(str(class_name)), [Point32(x1, y1, 0.0), Point32(x2, y2, 0.0)], [], Int32(image_index)))
                    indices.append(i)

                    #self.object_info[class_name].append(cv_image) # Store image // Igor
                    
                    image_path_combination += f'{class_name}_'
                    object_detected = True
        if len(self.image_queue) >= self.image_queue_size:
            self.save_image()  # Save the image if the queue is full
        if object_detected:
            self.image_queue.append((start_time, cv_image, object_array_for_queue, object_class_names_array))  # Store the image with timestamp and class name
        #Igor
        """
        if object_detected:
            current_time = time.time()
            if (self.last_class_name != image_path_combination) or (current_time - self.last_save_time > 3.0):
                cv2.imwrite(f'{self.image_path}{image_path_combination}{self.image_index}.jpg', cv_image)
                object_detected = False
                self.last_class_name = image_path_combination
                self.last_save_time = current_time
                self.image_index += 1
        """
        self.last_odom_position = odom_msg.pose.pose.position
        self.last_odom_orientation = odom_msg.pose.pose.orientation


        h2, w2, _ = results[0].orig_img.shape
        masks = None

        # Define range of brightness in HSV
        lower_black = np.array([0,0,0])
        upper_black = np.array([0,0,1])
        
        # print(time.time() - start_time)
        rospy.loginfo(f'[TIMIMG]: {time.time() - start_time}')
        
        n = 0
        for i in indices:
            mask = results[0].masks[i]
            
            segment = []

            for point in mask.xy[0]:
                segment.append(Point32(int(point[0]), int(point[1]), 0.0))
                
            detected_objects[n].segment = segment
            n += 1
            
            mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
            
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)

            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
            mask = cv2.resize(mask_3channel, (w2, h2))

            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)

            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)

            if masks is None:
                masks = mask
            else:
                masks = cv2.bitwise_or(mask, masks)
        
        # Apply the mask to the original image
        masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=masks)
                
        self.segmented_image_pub.publish(self.bridge.cv2_to_imgmsg(masked, encoding='bgr8'))
        
        detected_objects_msg = DetectedObjects()
        detected_objects_msg.objects = detected_objects
        detected_objects_msg.header.stamp = rospy.Time.now()
        depth_msg.header.stamp = rospy.Time.now()
        self.rgb_image.header.stamp = rospy.Time.now()
        odom_msg.header.stamp = rospy.Time.now()
        
        self.rgb_image.header.frame_id = 'map'
        depth_msg.header.frame_id = 'map'
        odom_msg.header.frame_id = 'map'
        
        print('depth: ', depth_msg.header.stamp.nsecs)
        print('image: ', self.rgb_image.header.stamp.nsecs)
        print('odom: ', odom_msg.header.stamp.nsecs)
        print('detected objects: ', detected_objects_msg.header.stamp.nsecs)
        
        # publish depth image and odometry together with segmented image for synchronization
        self.image_pub.publish(self.rgb_image)
        self.depth_pub.publish(depth_msg)
        self.odom_pub.publish(odom_msg)
        self.detected_objects_pub.publish(detected_objects_msg)
        
    def save_image(self):
        if not self.queue_saved_empty:

            for entry in reversed(self.image_queue_saved):
                last_saved_time, last_saved_class_names = entry
                last_saved_counts = Counter(last_saved_class_names)
                self.save_image_helper_saved(last_saved_time, last_saved_counts)

        if len(self.image_queue) == 0:
            return
        start_time, cv_image, object_array, object_class_names_array = self.image_queue.popleft() #self.image_queue[0]
        current_counts = Counter(object_class_names_array)
        
        for entry in reversed(self.image_queue):
            if  entry[0] - start_time > self.comparison_time_range:
                break
            entry_object_class_names_array = entry[3]
            entry_object_counts = Counter(entry_object_class_names_array)

            # Check if the current image is equal or a subset of the entry image
            if current_counts == entry_object_counts:
                return  # Skip saving this image
            for class_name in current_counts:
                if class_name not in entry_object_counts or current_counts[class_name] > entry_object_counts[class_name]:
                    continue
        


        # save the image
        image_path_combination = ''
        for detected_object_name in object_class_names_array:
            image_path_combination += detected_object_name + '_'
            
        cv2.imwrite(f'{self.image_path}{image_path_combination}{self.image_index}.jpg', cv_image)
        self.last_class_name = image_path_combination
        self.last_save_time = start_time
        self.image_index += 1
        self.image_queue_saved.append((start_time, object_class_names_array))
        self.queue_saved_empty = False
        
    def save_image_helper_saved(self, last_saved_time, last_saved_counts):    
        while (len(self.image_queue) > 0):
                start_time_ = self.image_queue[0][0]
                object_class_names_array_ = self.image_queue[0][3]
                current_counts = Counter(object_class_names_array_)
                #print(f'last saved time: {last_saved_time}, start time: {start_time_}, current counts: {current_counts}, last saved counts: {last_saved_counts}')
                if start_time_ - last_saved_time > self.comparison_time_range:
                    return
                # Check if the current image is equal or a subset of the last saved image
                if current_counts == last_saved_counts:
                    self.image_queue.popleft()
                    continue
                for class_name in current_counts:
                    if class_name not in last_saved_counts or current_counts[class_name] > last_saved_counts[class_name]:
                        return
                
                # If all classes are equal or a subset, skip saving this image
                self.image_queue.popleft()
                continue

    def run(self):
        # Keep the node running
        rospy.spin()
    
    #Igor
    #def calculate_time_formula(self):


if __name__ == '__main__':
    try:
        node = YOLOv9SegNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
