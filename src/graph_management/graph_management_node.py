#!/usr/bin/env python3

import rospy
import networkx as nx
import numpy as np
import math
import json
from scene_graph.msg import GraphObject, GraphObjects, ClassifiedRoom, RoomPolygonList, RoomWithObjects
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32
from tf.transformations import quaternion_from_matrix
from std_msgs.msg import String, Int32, Bool
import pickle
import time
from shapely.geometry import Polygon


class BuildingNode:
    def __init__(self, id, center_point) -> None:
        self.id = id
        self.center_point = center_point

class RoomNode:
    def __init__(self, id, class_id, polygon, center_point) -> None:
        self.id = id
        self.class_id = class_id
        self.polygon = polygon
        self.center_point = center_point

class ObjectNode:
    def __init__(self, id, class_id, bounding_box, object_name=None) -> None:
        self.id = id
        self.class_id = class_id
        self.bounding_box = bounding_box
        self.object_name = object_name if object_name else class_id
        #self.image_index = image_index  # Igor
        #self.description = description  # Igor

class GraphManagementNode:

    def __init__(self) -> None:
        rospy.Subscriber('/scene_graph/seen_graph_objects', GraphObjects, self.seen_objects_callback)
        rospy.Subscriber('/scene_graph/rooms', RoomPolygonList, self.rooms_callback)
        rospy.Subscriber('/scene_graph/classified_room', ClassifiedRoom, self.classified_room_callback)
        rospy.Subscriber('/scene_graph/control', Bool, self.control_callback)
        rospy.Subscriber('/scene_graph/object_relationships', String, self.relationships_callback)
        
        self.objects_pub = rospy.Publisher('scene_graph/graph_objects', GraphObjects, queue_size=10)
        self.room_with_objects_pub = rospy.Publisher('/scene/graph/room_with_objects', RoomWithObjects, queue_size=10)
        
        # marker pubs
        self.object_bbox_markers_pub = rospy.Publisher('/scene_graph/viz/object_bbox_marker', MarkerArray, queue_size=10)
        self.points_marker_pub = rospy.Publisher('/scene_graph/debug/bbox_points', MarkerArray, queue_size=10)
        self.room_markers_pub = rospy.Publisher('/scene_graph/viz/room_markers', MarkerArray, queue_size=10)
        self.building_markers_pub = rospy.Publisher('/scene_graph/viz/building_markers', MarkerArray, queue_size=10)
        self.line_markers_pub = rospy.Publisher('/scene_graph/viz/line_markers', MarkerArray, queue_size=10)
        self.text_markers_pub = rospy.Publisher('/scene_graph/viz/text_markers', MarkerArray, queue_size=10)
        self.relationship_markers_pub = rospy.Publisher('/scene_graph/viz/relationship_markers', MarkerArray, queue_size=10)
        
        self.scene_graph = nx.Graph()
        
        self.scene_graph.add_node(0, data=BuildingNode(0, (0.0, 0.0)))
        self.scene_graph.add_node(1, data=RoomNode(1, 'default_room', None, (0.0, 0.0)))
        self.scene_graph.add_edge(0, 1)
        
        self.current_iteration = 1
        
        self.n = 2
        
        self.graph_lock = False
        
        self.rooms_segmented = False
        
        # id, size
        self.rooms = [[1, 999]]
        
        # room where the robot currently is in
        self.current_room_id = 1
        self.rooms_classified = False
        
        self.overlapping_threshold = 0.8
        
        self.old_marker_ids = []
        
        # Object relationships from spatial tracker
        self.object_relationships = {}  # object_id -> {related_object_id: relationship_description}
        
        # Mapping from spatial tracker object IDs to graph node IDs
        self.spatial_to_graph_id_map = {}  # spatial_object_id -> graph_node_id
        self.graph_to_spatial_id_map = {}  # graph_node_id -> spatial_object_id


    def run_main_loop(self):
        while not rospy.is_shutdown():
            
            self.graph_lock = True
            self.publish_markers()
            self.graph_lock = False

            nodes = list(self.scene_graph.nodes)
            # Find room, extract objects in room and publish //Igor
            for node in nodes:
                if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                    if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    
                        object_names_in_room = []
                        children = list(self.scene_graph.neighbors(node))

                        for c in children:
                            if c in self.scene_graph and 'data' in self.scene_graph.nodes[c]:
                                if type(self.scene_graph.nodes[c]['data']) is ObjectNode:
                                    object_names_in_room.append(String(self.scene_graph.nodes[c]['data'].class_id))
                                        
                                    
                        room_with_objects_msg = RoomWithObjects()
                        room_with_objects_msg.header.stamp = rospy.Time.now()
                        room_with_objects_msg.id = Int32(node)
                        room_with_objects_msg.objects = object_names_in_room
                        
                        self.room_with_objects_pub.publish(room_with_objects_msg)


            while self.graph_lock:
                time.sleep(0.01)

            self.graph_lock = True
            nodes = list(self.scene_graph.nodes)
            # Redo edges for objects and rooms ==> find adjacent Rooms //Igor    
            self.scene_graph.remove_edges_from(list(self.scene_graph.edges))
            for object_node in nodes:
                if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
                    if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
                        min = np.inf
                        min_node = 0
                        for room_node in nodes:
                            if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                                if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
                                    
                                    if (self.scene_graph.nodes[room_node]['data'].polygon is None):
                                        continue
                                    
                                    if (self.is_object_in_room(self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']),  self.scene_graph.nodes[room_node]['data'].polygon)):
                                        # print(f'object {object_node} is in room {room_node}')
                                        object_id = self.scene_graph.nodes[object_node]['data'].id
                                        self.scene_graph.add_edge(room_node, object_id)
                                        break
                        
                    elif type(self.scene_graph.nodes[object_node]['data']) is RoomNode:
                        
                        for room_node in nodes:
                            if type(self.scene_graph.nodes[room_node]['data']) is RoomNode and room_node != object_node:
                                if self.has_adjacent_points(self.scene_graph.nodes[object_node]['data'].polygon, self.scene_graph.nodes[room_node]['data'].polygon):
                                    self.scene_graph.add_edge(room_node, object_node)

            # Identify unconnected objects and add edges to closest room //Igor    
            for object_node in list(self.scene_graph.nodes):
                if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
                    if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:     
                        
                        has_edge = False     
                        for edge in list(self.scene_graph.edges):
                            if edge[0] == object_node or edge[1] == object_node:
                                has_edge = True
                                
                        if not has_edge:
                            
                            print(object_node, self.scene_graph.nodes[object_node]['data'].class_id, 'has no edge')
                            min = np.inf
                            min_node = 0
                            
                            for room_node in list(self.scene_graph.nodes):
                                if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                                    if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
                                        if self.euclidean_distance2d(self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), self.scene_graph.nodes[room_node]['data'].center_point) < min:
                                            min = self.euclidean_distance2d(self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), self.scene_graph.nodes[room_node]['data'].center_point)
                                            min_node = room_node
                                        
                            self.scene_graph.add_edge(min_node, object_node)
                            print('adding edge', object_node, min_node)
                        

            self.publish_markers()

            self.graph_lock = False
            rospy.rostime.wallsleep(0.5)
        
        
    # save and delete graph
    def control_callback(self, msg):
        if msg.data == True:
            
            time.sleep(5.0)
            
            pickle.dump(self.scene_graph, open(f"scene_graph_{self.current_iteration}.pkl", 'wb'))
            
            self.current_iteration += 1
            
            self.scene_graph = nx.Graph()
        
            self.scene_graph.add_node(0, data=BuildingNode(0, (0.0, 0.0)))
            self.scene_graph.add_node(1, data=RoomNode(1, 'default_room', None, (0.0, 0.0)))
            self.scene_graph.add_edge(0, 1)
            
            self.graph_lock = False
        
            self.current_room_id = 1
            self.rooms_classified = False
            
            self.rooms_segmented = False
            self.rooms = [[1, 999]]
            
            # Clear relationship mappings
            self.object_relationships.clear()
            self.spatial_to_graph_id_map.clear()
            self.graph_to_spatial_id_map.clear()
    
    def relationships_callback(self, msg):
        """Handle object relationships from spatial tracker"""
        try:
            # Parse JSON string containing relationships
            relationships_data = json.loads(msg.data)
            self.object_relationships = relationships_data
            
            rospy.loginfo(f"Received relationships for {len(relationships_data)} objects")
            rospy.logdebug(f"Relationships: {relationships_data}")
            
        except json.JSONDecodeError as e:
            rospy.logerr(f"Failed to parse relationships JSON: {e}")
        except Exception as e:
            rospy.logerr(f"Error processing relationships: {e}")
    
    def get_object_id_from_name(self, object_name):
        """Find graph node ID from object name (for cases where we don't have spatial ID mapping)"""
        nodes = list(self.scene_graph.nodes)
        for node_id in nodes:
            node_data = self.scene_graph.nodes[node_id].get('data')
            if isinstance(node_data, ObjectNode):
                if node_data.object_name == object_name or node_data.class_id == object_name:
                    return node_id
        return None
    
    def update_relationship_edges(self):
        """Add edges between objects based on relationship data from spatial tracker"""
        if not self.object_relationships:
            return
        
        relationship_edges_added = 0
        
        for spatial_obj_id_str, relationships in self.object_relationships.items():
            try:
                spatial_obj_id = int(spatial_obj_id_str)
                
                # Get graph node ID for this spatial object
                source_graph_id = self.spatial_to_graph_id_map.get(spatial_obj_id)
                if not source_graph_id:
                    continue
                
                # Add edges to related objects
                for related_spatial_id_str, relationship_desc in relationships.items():
                    try:
                        related_spatial_id = int(related_spatial_id_str)
                        target_graph_id = self.spatial_to_graph_id_map.get(related_spatial_id)
                        
                        if target_graph_id:
                            # Add edge with relationship data as attribute
                            if not self.scene_graph.has_edge(source_graph_id, target_graph_id):
                                self.scene_graph.add_edge(source_graph_id, target_graph_id, 
                                                        relationship=relationship_desc,
                                                        edge_type='object_relationship')
                                relationship_edges_added += 1
                                rospy.logdebug(f"Added relationship edge: {source_graph_id} -> {target_graph_id} ({relationship_desc})")
                    except ValueError:
                        continue
                        
            except ValueError:
                continue
        
        if relationship_edges_added > 0:
            rospy.loginfo(f"Added {relationship_edges_added} relationship edges to scene graph")
    
    def create_relationship_marker(self, point1, point2, relationship_desc, marker_id):
        """Create a line marker to visualize object relationships"""
        # Create line from object centers
        center1 = self.calculate_bounding_box_center([point1[0], point1[1]])
        center2 = self.calculate_bounding_box_center([point2[0], point2[1]])
        
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "relationships"
        line_marker.id = marker_id
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.08  # Thicker line for relationships
        line_marker.lifetime = rospy.Duration(0.6)

        # Color based on relationship type
        if "part" in relationship_desc.lower():
            # Blue for part relationships
            line_marker.color.r = 0.0
            line_marker.color.g = 0.4
            line_marker.color.b = 1.0
        elif "nearby" in relationship_desc.lower() or "spatial" in relationship_desc.lower():
            # Green for spatial relationships
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
        else:
            # Orange for other relationships
            line_marker.color.r = 1.0
            line_marker.color.g = 0.5
            line_marker.color.b = 0.0
        
        line_marker.color.a = 0.8
        
        line_marker.pose.orientation.x = 0.0
        line_marker.pose.orientation.y = 0.0
        line_marker.pose.orientation.z = 0.0
        line_marker.pose.orientation.w = 1.0

        # Add points at slightly elevated Z to avoid ground collision
        z_offset = 0.5
        point1_3d = Point32(center1.x, center1.y, center1.z + z_offset)
        point2_3d = Point32(center2.x, center2.y, center2.z + z_offset)
        
        line_marker.points.append(point1_3d)
        line_marker.points.append(point2_3d)
        
        return line_marker
    
    def create_relationship_text_marker(self, point1, point2, relationship_desc, marker_id):
        """Create a text marker to show relationship description"""
        # Position text at midpoint between objects
        center1 = self.calculate_bounding_box_center([point1[0], point1[1]])
        center2 = self.calculate_bounding_box_center([point2[0], point2[1]])
        
        mid_x = (center1.x + center2.x) / 2.0
        mid_y = (center1.y + center2.y) / 2.0
        mid_z = (center1.z + center2.z) / 2.0 + 1.0  # Elevated text
        
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "relationship_labels"
        text_marker.id = marker_id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(0.6)
        text_marker.pose.position = Point32(mid_x, mid_y, mid_z)
        text_marker.scale.z = 0.3  # Smaller text for relationships

        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 0.9

        text_marker.text = relationship_desc
        
        return text_marker
    
    
    def classified_room_callback(self, msg):
        nodes = list(self.scene_graph.nodes)
        
        for room in self.rooms:
            if self.scene_graph.nodes[room[0]]['data'].id == msg.id.data:
                self.scene_graph.nodes[room[0]]['data'].class_id = msg.label.data
                break
        
    # Check if object is in graph, if yes merge bounding boxes , if no add object to room
    def seen_objects_callback(self, msg):
        
        # to disable interrupting the constructor
        if len(list(self.scene_graph.nodes)) < 2:
            return
        
        if self.graph_lock:
            time.sleep(0.1)
        
        self.graph_lock = True
        
        start_time = time.time()
        
        changed_room_ids = []
        
        for object in msg.objects: #
            
            # Extract spatial object ID from the GraphObject
            spatial_object_id = object.object_id.data
            
            in_graph = self.is_object_in_graph(object)
            
            if in_graph == -1:

                self.scene_graph.add_node(self.n, data=ObjectNode(self.n, object.name.data, object.bounding_box, object.name.data))
                
                # Map spatial object ID to graph node ID
                self.spatial_to_graph_id_map[spatial_object_id] = self.n
                self.graph_to_spatial_id_map[self.n] = spatial_object_id

                if not self.rooms_classified:
                    
                    # add to default room first
                    self.scene_graph.add_edge(self.current_room_id, self.n)
                else:
                    nodes = list(self.scene_graph.nodes)
                    for room in self.rooms:
                        
                            if self.is_object_in_room(self.get_object_2d_position(object), self.scene_graph.nodes[room[0]]['data'].polygon):
                                self.scene_graph.add_edge(self.scene_graph.nodes[room[0]]['data'].id, self.n)
                                
                                if not self.scene_graph.nodes[room[0]]['data'].id in changed_room_ids:
                                    changed_room_ids.append(self.scene_graph.nodes[room[0]]['data'].id)
                                break
                
                self.n += 1   
                
            else:
                # Update spatial object ID mapping for existing object
                self.spatial_to_graph_id_map[spatial_object_id] = in_graph
                self.graph_to_spatial_id_map[in_graph] = spatial_object_id
                
                merged_box = self.merge_bounding_boxes(self.scene_graph.nodes[in_graph]['data'].bounding_box[0], self.scene_graph.nodes[in_graph]['data'].bounding_box[1],
                                          object.bounding_box[0], object.bounding_box[1])
                
                self.scene_graph.nodes[in_graph]['data'].bounding_box[0] = merged_box[0]
                self.scene_graph.nodes[in_graph]['data'].bounding_box[1] = merged_box[1]
        
        # Process object relationships after all objects are added/updated
        self.update_relationship_edges()

        rospy.loginfo(f'[NODES]: {len(self.scene_graph.nodes)}')
        rospy.loginfo(f'[TIMIMG]: {time.time() - start_time}')

        self.graph_lock = False


    def publish_markers(self):
        
        # publishing of markers
        # TODO: use less loops!
        
        # publish bounding box markers for each object
        marker_array = MarkerArray()
                
        bounding_boxes = []
        nodes = list(self.scene_graph.nodes)
        
        # print()
        
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    
                    bounding_boxes.append(
                            [
                                self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[0]),
                                self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[1]),
                            ]
                    )
                
        for i, (point1, point2) in enumerate(bounding_boxes):
            marker = self.create_marker_from_bbox(point1, point2, i)
            marker_array.markers.append(marker)
        
        graph_objects_msg = GraphObjects()
        graph_objects_msg.header.stamp = rospy.Time.now()
        
        graph_objects = []
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    # Get spatial tracker object ID if available, otherwise use graph node ID
                    spatial_obj_id = self.graph_to_spatial_id_map.get(node, node)
                    graph_objects.append(GraphObject(String(self.scene_graph.nodes[node]['data'].class_id), Int32(spatial_obj_id), self.scene_graph.nodes[node]['data'].bounding_box))#, Int32(self.scene_graph.nodes[node]['data'].image_index))) #Igor

        graph_objects_msg.objects = graph_objects
        
        self.objects_pub.publish(graph_objects_msg)
        
        # publish building markers
        building_maker_array = MarkerArray()
        
        for i, node in enumerate(nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    building_maker_array.markers.append(self.create_building_marker(self.scene_graph.nodes[node]['data'].center_point, i))
        
        delete_marker_array = MarkerArray()
        for i in self.old_marker_ids:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "room_markers"
            marker.id = i  # Unique ID for each marker
            marker.action = Marker.DELETE  # Set action to DELETE for all markers
            marker_array.markers.append(marker)

        self.room_markers_pub.publish(delete_marker_array)
        
        self.old_marker_ids = []
        
        # publish room markers
        room_marker_array = MarkerArray()
        
        for i, node in enumerate(nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    room_marker_array.markers.append(self.create_room_marker(self.scene_graph.nodes[node]['data'].polygon, i))
                    self.old_marker_ids.append(i)
        
        
        # publish line markers
        line_marker_array = MarkerArray()
        
        object_ids = []
        
        # first building level to room level
        i = 0
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    
                    # TODO building center and room center have to be 3d!
                    building_center_point = (self.scene_graph.nodes[0]['data'].center_point[0], self.scene_graph.nodes[0]['data'].center_point[1], 15.0)
                    room_center_point = (self.scene_graph.nodes[node]['data'].center_point[0], self.scene_graph.nodes[node]['data'].center_point[1], 8.0)
                    
                    line_marker_array.markers.append(self.create_line_marker(building_center_point, room_center_point, i))
                    i += 1
                    
                    target_objects = []
                    
                    # get object nodes
                    for edge in list(self.scene_graph.edges):
                        
                        if self.scene_graph.nodes[node]['data'].id == edge[0]: 
                            target_objects.append(edge[1])
                            
                        elif self.scene_graph.nodes[node]['data'].id == edge[1]:
                            target_objects.append(edge[0])
                            
                            
                    for target in target_objects:
                        if target in self.scene_graph and 'data' in self.scene_graph.nodes[target]:
                            if type(self.scene_graph.nodes[target]['data']) == BuildingNode or type(self.scene_graph.nodes[target]['data']) == RoomNode:
                                continue
                            
                            # TODO
                            target_center_point = self.calculate_bounding_box_center(self.scene_graph.nodes[target]['data'].bounding_box)#
                            target_center_point = (target_center_point.x, target_center_point.y, target_center_point.z)
                            
                            line_marker_array.markers.append(self.create_line_marker(self.scene_graph.nodes[node]['data'].center_point, target_center_point, i))
                            i += 1
                    
        for edge in list(self.scene_graph.edges):
            if type(self.scene_graph.nodes[edge[0]]['data']) == RoomNode and type(self.scene_graph.nodes[edge[1]]['data']) == RoomNode:
                room_center_point_1 = (self.scene_graph.nodes[edge[0]]['data'].center_point[0], self.scene_graph.nodes[edge[0]]['data'].center_point[1], 8.0)
                room_center_point_2 = (self.scene_graph.nodes[edge[1]]['data'].center_point[0], self.scene_graph.nodes[edge[1]]['data'].center_point[1], 8.0)
                
                line_marker_array.markers.append(self.create_line_marker(room_center_point_1, room_center_point_2, i))
                i += 1
                                         
        
        
        # publish text markers
        text_marker_array = MarkerArray()
        
        id = 0
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    position = (self.scene_graph.nodes[node]['data'].center_point[0], self.scene_graph.nodes[node]['data'].center_point[1], 15.0)
                    text_marker_array.markers.append(self.create_text_marker(position, "building", id))
                    id += 1

                elif type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    position = (self.scene_graph.nodes[node]['data'].center_point[0], self.scene_graph.nodes[node]['data'].center_point[1], 8.0)
                    text_marker_array.markers.append(self.create_text_marker(position, self.scene_graph.nodes[node]['data'].class_id, id))
                    id += 1
                    
                elif type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    position = self.calculate_bounding_box_center(self.scene_graph.nodes[node]['data'].bounding_box)
                    position = (position.x, position.y, position.z)
                    text_marker_array.markers.append(self.create_text_marker(position, self.scene_graph.nodes[node]['data'].class_id, id))
                    id += 1
            
            
        # delete old marker
            
        self.object_bbox_markers_pub.publish(marker_array)
        self.objects_pub.publish(graph_objects_msg)
        self.building_markers_pub.publish(building_maker_array)
        self.room_markers_pub.publish(room_marker_array)
        self.line_markers_pub.publish(line_marker_array)
        self.text_markers_pub.publish(text_marker_array)
        
        # Publish relationship markers
        self.publish_relationship_markers()
        
        # save graph in file
        # pickle.dump(self.scene_graph, open('/home/nes/catkin_ws/src/scene_graph/src/graph_creation/scene_graph.pickle', 'wb'))
        
        # self.graph_lock = False
    
    def publish_relationship_markers(self):
        """Publish visualization markers for object relationships"""
        relationship_marker_array = MarkerArray()
        relationship_text_array = MarkerArray()
        
        marker_id = 0
        
        # Get all edges that represent object relationships
        for edge in self.scene_graph.edges(data=True):
            source_node, target_node, edge_data = edge
            
            # Only process relationship edges between objects
            if edge_data.get('edge_type') == 'object_relationship':
                source_data = self.scene_graph.nodes[source_node].get('data')
                target_data = self.scene_graph.nodes[target_node].get('data')
                
                # Make sure both are object nodes (not room or building nodes)
                if (isinstance(source_data, ObjectNode) and isinstance(target_data, ObjectNode)):
                    relationship_desc = edge_data.get('relationship', 'related')
                    
                    # Create line marker for relationship
                    line_marker = self.create_relationship_marker(
                        source_data.bounding_box, target_data.bounding_box, 
                        relationship_desc, marker_id
                    )
                    relationship_marker_array.markers.append(line_marker)
                    
                    # Create text marker for relationship description
                    text_marker = self.create_relationship_text_marker(
                        source_data.bounding_box, target_data.bounding_box,
                        relationship_desc, marker_id + 1000  # Offset for text markers
                    )
                    relationship_text_array.markers.append(text_marker)
                    
                    marker_id += 1
        
        # Publish the relationship markers
        self.relationship_markers_pub.publish(relationship_marker_array)
        
        # Also add relationship texts to the main text marker publisher
        for text_marker in relationship_text_array.markers:
            text_marker.ns = "relationship_texts"
        self.text_markers_pub.publish(relationship_text_array)
        
        rospy.logdebug(f"Published {len(relationship_marker_array.markers)} relationship markers")
               
        
    def merge_bounding_boxes(self, min1, max1, min2, max2):
        """
        Merges two axis-aligned 3D bounding boxes.

        Parameters:
        - min1: tuple of (x1_min, y1_min, z1_min) for the first bounding box
        - max1: tuple of (x1_max, y1_max, z1_max) for the first bounding box
        - min2: tuple of (x2_min, y2_min, z2_min) for the second bounding box
        - max2: tuple of (x2_max, y2_max, z2_max) for the second bounding box

        Returns:
        - merged_min: tuple of (x_min, y_min, z_min) for the merged bounding box
        - merged_max: tuple of (x_max, y_max, z_max) for the merged bounding box
        """
        
        min1 = (min1.x, min1.y, min1.z)
        min2 = (min2.x, min2.y, min2.z)
        max1 = (max1.x, max1.y, max1.z)
        max2 = (max2.x, max2.y, max2.z)
        
        # Calculate the merged bounding box minimum coordinates
        merged_min = (
            min(min1[0], min2[0]),  # x_min
            min(min1[1], min2[1]),  # y_min
            min(min1[2], min2[2])   # z_min
        )

        # Calculate the merged bounding box maximum coordinates
        merged_max = (
            max(max1[0], max2[0]),  # x_max
            max(max1[1], max2[1]),  # y_max
            max(max1[2], max2[2])   # z_max
        )
        
        merged_min = Point32(merged_min[0], merged_min[1], merged_min[2])
        merged_max = Point32(merged_max[0], merged_max[1], merged_max[2])

        return merged_min, merged_max
        
        
    def create_point_marker_from_bbox(self, point, id):
        marker = Marker()
    
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.a = 0.7  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 0.0  # Red color (Change as needed)
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker
    
    
    def create_building_marker(self, center_point, id):
        marker = Marker()
        
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.6)
        
        marker.pose.position.x = center_point[0]
        marker.pose.position.y = center_point[1]
        marker.pose.position.z = 15.0
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.a = 1.0  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 0.25  # Red color (Change as needed)
        marker.color.g = 0.60
        marker.color.b = 0.84

        return marker
    
    
    def create_text_marker(self, position, text, id):
        text_marker = Marker()
        text_marker.header.frame_id = "map"  # or any frame you're using
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "labels"
        text_marker.id = id  # Ensure unique IDs for all markers
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(0.6)
        text_marker.pose.position = Point32(
            position[0],
            position[1],
            position[2] + 0.8  # Slightly above 
        )
        text_marker.scale.z = 0.7  # Text size

        # Properly initialize the orientation quaternion for the text marker
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        # Define the color of the text
        text_marker.color.r = 0.25  # Red color (Change as needed)
        text_marker.color.g = 0.60
        text_marker.color.b = 0.84
        text_marker.color.a = 1.0  # Alpha (transparency)

        # Set the text content
        text_marker.text = text
        
        return text_marker
    
    
    def create_room_marker(self, polygon, id):
        marker = Marker()
        
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "room_markers"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.6)
        
        center_x = 0.0
        center_y = 0.0
        
        if polygon is not None:
            for point in polygon:
                center_x += point.x
                center_y += point.y
                
            center_x /= len(polygon)
            center_y /= len(polygon)

        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = 8.0
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.a = 1.0  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 0.25  # Red color (Change as needed)
        marker.color.g = 0.60
        marker.color.b = 0.84

        return marker
    
    
    def create_line_marker(self, point1, point2, i):
        
        z1 = 8.0
        z2 = 8.0
        
        if len(point1) > 2:
            z1 = point1[2]
        
        if len(point2) > 2:
            z2 = point2[2]
        
        point1 = Point32(point1[0], point1[1], z1)
        point2 = Point32(point2[0], point2[1], z2)
        
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "lines"
        line_marker.id = i
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.04
        line_marker.lifetime = rospy.Duration(0.6)

        line_marker.color.r = 0.25  # Red color (Change as needed)
        line_marker.color.g = 0.60
        line_marker.color.b = 0.84
        line_marker.color.a = 0.7
        
        line_marker.pose.orientation.x = 0.0
        line_marker.pose.orientation.y = 0.0
        line_marker.pose.orientation.z = 0.0
        line_marker.pose.orientation.w = 1.0

        line_marker.points.append(point1)
        line_marker.points.append(point2)
        
        return line_marker
    
            
    def create_marker_from_bbox(self, point1, point2, id):
        marker = Marker()
    
        # Calculate center
        center_x = (point1[0] + point2[0]) / 2.0
        center_y = (point1[1] + point2[1]) / 2.0
        center_z = (point1[2] + point2[2]) / 2.0

        # Calculate dimensions
        size_x = abs(point1[0] - point2[0])
        size_y = abs(point1[1] - point2[1])
        size_z = abs(point1[2] - point2[2])

        # Assume we have two vectors along the edges of the box
        v1 = np.array([point2[0] - point1[0], 0, 0])
        v2 = np.array([0, point2[1] - point1[1], 0])
        v3 = np.array([0, 0, point2[2] - point1[2]])

        # Create orthonormal basis
        x_axis = v1 / np.linalg.norm(v1)
        y_axis = v2 / np.linalg.norm(v2)
        z_axis = v3 / np.linalg.norm(v3)

        # Construct rotation matrix (3x3)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

        # Create homogeneous transformation matrix (4x4)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        # Convert rotation matrix to quaternion
        quaternion = quaternion_from_matrix(transformation_matrix)
        
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_boxes"
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.6)

        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        marker.scale.x = size_x
        marker.scale.y = size_y
        marker.scale.z = size_z

        marker.color.a = 0.5  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 1.0  # Red color (Change as needed)
        marker.color.g = 0.0
        marker.color.b = 0.0

        return marker   
    
      
    # TODO: implement  
    def rooms_callback(self, msg):
        # go to (all) objects in graph
        # check if object is in room
        # create sub graph for each room
        # add sub graph to graph
        
        while self.graph_lock:
            time.sleep(0.01)
            
        self.graph_lock = True
        
        for node in list(self.scene_graph.nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    self.scene_graph.remove_node(node)
            
        self.rooms = []
        self.rooms_classified = True
                
        self.scene_graph.remove_edges_from(list(self.scene_graph.edges))
             
        for room in msg.rooms:
            
            center_x = 0.0
            center_y = 0.0
        
            if room.points is not None:
                for point in room.points:
                    center_x += point.x
                    center_y += point.y
                    
                center_x /= len(room.points)
                center_y /= len(room.points)
                
            self.scene_graph.add_node(self.n, data=RoomNode(self.n, '-', room.points, (center_x, center_y)))
            self.scene_graph.add_edge(0, self.n)
            self.rooms.append([self.n, self.polygon_area(room.points)])
            self.n += 1
                
        # sort rooms list
        self.rooms = sorted(self.rooms, key=lambda x: x[1])
        
        # calculate building center
        center_x = 0.0
        center_y = 0.0
        
        room_count = 0
        nodes = list(self.scene_graph.nodes)
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    center_x += self.scene_graph.nodes[node]['data'].center_point[0]
                    center_y += self.scene_graph.nodes[node]['data'].center_point[1]
                    room_count += 1
                
        center_x /= room_count
        center_y /= room_count
        
        self.scene_graph.nodes[0]['data'].center_point = (center_x, center_y)
        
        
        # TODO: make it smarter
        # redo all edges
        nodes = list(self.scene_graph.nodes)
                
        for object_node in nodes:
            if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
                if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
                    print('processing object', object_node)
                    min = np.inf
                    min_node = 0
                    for room_node in nodes:
                        if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                            if type(self.scene_graph.nodes[object_node]['data']) is RoomNode:
                                if self.euclidean_distance2d(self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), self.scene_graph.nodes[room_node]['data'].center) < min:
                                    min = self.euclidean_distance2d(self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), self.scene_graph.nodes[room_node]['data'].center)
                                    min_node = room_node
                                
                    self.scene_graph.add_edge(min_node, object_node)
                        
        self.rooms_segmented = True
        self.graph_lock = False
        
        
        
    def is_object_in_room(self, point, vertices):
        """
        Determine if a point is inside a given polygon or not.

        Args:
        point: A tuple (x, y) representing the coordinates of the point to check.
        polygon: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the vertices of the polygon.

        Returns:
        True if the point is inside the polygon, False otherwise.
        """
        # x, y = point
        # n = len(polygon)
        # inside = False
        
        # min_x = np.inf
        # min_y = np.inf
        # max_x = -np.inf
        # max_y = -np.inf
        
        # for point in polygon:
        #     if point.x < min_x:
        #         min_x = point.x
        #     if point.y < min_y:
        #         min_y = point.y
        #     if point.x > max_x:
        #         max_x = point.x
        #     if point.y > max_y:
        #         max_y = point.y
                
        # print(min_x, max_x, min_y, max_y)
                
        # if min_x < x < max_x and min_y < y < max_y:
        #     return True
        # else:
        #     return False

        # p1x, p1y = polygon[0]
        # p1x = polygon[0].x
        # p1y = polygon[0].y
        # for i in range(n + 1):
        #     # p2x, p2y = polygon[i % n]
            
        #     p2x = polygon[i % n].x
        #     p2y = polygon[i % n].y
            
        #     if y > min(p1y, p2y):
        #         if y <= max(p1y, p2y):
        #             if x <= max(p1x, p2x):
        #                 if p1y != p2y:
        #                     xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        #                 if p1x == p2x or x <= xinters:
        #                     inside = not inside
        #     p1x, p1y = p2x, p2y

        # return inside
        
        point = Point32(point[0], point[1], 0.0)
        
        winding_number = 0
        n = len(vertices)
        for i in range(n):
            V_i = vertices[i]
            V_next = vertices[(i + 1) % n]

            if V_i.y <= point.y:
                if V_next.y > point.y:
                    is_left = ((V_next.x - V_i.x) * (point.y - V_i.y) -
                            (point.x - V_i.x) * (V_next.y - V_i.y))
                    if is_left > 0:
                        winding_number += 1
            else:
                if V_next.y <= point.y:
                    is_left = ((V_next.x - V_i.x) * (point.y - V_i.y) -
                            (point.x - V_i.x) * (V_next.y - V_i.y))
                    if is_left < 0:
                        winding_number -= 1
        return winding_number != 0
                    
    def get_object_2d_position(self, object):
        center_x = (object.bounding_box[0].x + object.bounding_box[1].x) / 2
        center_y = (object.bounding_box[0].y + object.bounding_box[1].y) / 2
        
        return (center_x, center_y)
        
    
    def is_object_in_graph(self, object):
        nodes = list(self.scene_graph.nodes)
        
        center_point = self.calculate_bounding_box_center(object.bounding_box)
        center_point = [center_point.x, center_point.y, center_point.z]
        
        for node in nodes:

            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
            
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                
                    overlap_volume = self.overlap_volume(object.bounding_box, self.scene_graph.nodes[node]['data'].bounding_box)
                    
                    if overlap_volume / self.bounding_box_volume(object.bounding_box) > self.overlapping_threshold or overlap_volume / self.bounding_box_volume(self.scene_graph.nodes[node]['data'].bounding_box) > self.overlapping_threshold:
                        return node
            
        return -1
    
    
    def is_point_inside_bbox(self, point1, point2, P):
        
        # Unpack the coordinates
        x1 = point1.x
        y1 = point1.y
        z1 = point1.z
        x2 = point2.x
        y2 = point2.y
        z2 = point2.z
        px = P.x
        py = P.y
        pz = P.z

        # Check if the point P is within the bounding box along each axis
        inside_x = min(x1, x2) <= px <= max(x1, x2)
        inside_y = min(y1, y2) <= py <= max(y1, y2)
        inside_z = min(z1, z2) <= pz <= max(z1, z2)

        # If the point is inside along all three axes, then it is inside the bounding box
        return inside_x and inside_y and inside_z
    
    
    def calculate_bounding_box_center(self, bounding_box):
        
        # Initialize sums for x, y, and z coordinates
        sum_x = sum_y = sum_z = 0.0
        
        # Sum all the coordinates
        for point in bounding_box:
            sum_x += point.x
            sum_y += point.y
            sum_z += point.z

        # Calculate the averages
        center_x = sum_x / 2
        center_y = sum_y / 2
        center_z = sum_z / 2

        return Point32(center_x, center_y, center_z)


    def euclidean_distance(self, point_1, point_2):
        return math.sqrt((point_2.x - point_1.x)**2 + (point_2.y - point_1.y)**2 + (point_2.z - point_1.z)**2)
    
    def euclidean_distance2d(self, point_1, point_2):
        return math.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)
    

    def tuple_from_point(self, point):
        return (point.x, point.y, point.z)
    
    
    def calculate_center(self, corners):
        """
        Calculate the center of the bounding box given its 8 corner points.
        
        Args:
            corners (list): A list of 8 corner points. Each point is a tuple (x, y, z).
        
        Returns:
            center (tuple): The center of the bounding box (x, y, z).
        """
        corners_np = np.array(corners)
        center = np.mean(corners_np, axis=0)
        return tuple(center)


    def calculate_orientation(self, corners):
        """
        Calculate the orientation of the bounding box given its 8 corner points.
        
        Args:
            corners (list): A list of 8 corner points. Each point is a tuple (x, y, z).
        
        Returns:
            quaternion (tuple): The orientation of the bounding box as a quaternion (x, y, z, w).
        """
        # Compute edge vectors
        vec_x = np.array(corners[1]) - np.array(corners[0])
        vec_y = np.array(corners[3]) - np.array(corners[0])
        vec_z = np.array(corners[4]) - np.array(corners[0])

        # Normalize vectors
        vec_x /= np.linalg.norm(vec_x)
        vec_y /= np.linalg.norm(vec_y)
        vec_z /= np.linalg.norm(vec_z)
        
        # Create a rotation matrix
        rotation_matrix = np.identity(4)  # 4x4 identity matrix
        rotation_matrix[0:3, 0] = vec_x  # Set x-axis
        rotation_matrix[0:3, 1] = vec_y  # Set y-axis
        rotation_matrix[0:3, 2] = vec_z  # Set z-axis
        
        # Convert the rotation matrix to a quaternion
        quaternion = quaternion_from_matrix(rotation_matrix)
        
        return self.normalize_quaternion(quaternion)
    
    
    def normalize_quaternion(self, quat):
        """
        Normalize a quaternion to ensure it is a unit quaternion.
        
        Args:
            quat (tuple): A quaternion (x, y, z, w).
            
        Returns:
            normalized_quat (tuple): The normalized quaternion (x, y, z, w).
        """
        norm = np.linalg.norm(quat)
        if norm == 0:
            rospy.logwarn("Zero norm quaternion, cannot normalize!")
            return (0, 0, 0, 1)
        
        return tuple(np.array(quat) / norm)


    def calculate_scale(self, corners):
        """
        Calculate the scale (dimensions) of the bounding box given its 8 corner points.
        
        Args:
            corners (list): A list of 8 corner points. Each point is a tuple (x, y, z).
        
        Returns:
            scale (tuple): The scale of the bounding box (scale_x, scale_y, scale_z).
        """
        # Compute distances between opposite corners to determine the dimensions
        scale_x = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))
        scale_y = np.linalg.norm(np.array(corners[3]) - np.array(corners[0]))
        scale_z = np.linalg.norm(np.array(corners[4]) - np.array(corners[0]))
        
        return (scale_x, scale_y, scale_z)
    
    
    def bounding_box_volume(self, bounding_box):
        """
        Calculate the volume of a 3D bounding box defined by two points.

        Parameters:
        - min_point: tuple of (x_min, y_min, z_min) for the bounding box
        - max_point: tuple of (x_max, y_max, z_max) for the bounding box

        Returns:
        - The volume of the bounding box.
        """
        min_point = [bounding_box[0].x, bounding_box[0].y, bounding_box[0].z]
        max_point = [bounding_box[1].x, bounding_box[1].y, bounding_box[1].z]
        
        # Calculate the length of each dimension
        length_x = max_point[0] - min_point[0]
        length_y = max_point[1] - min_point[1]
        length_z = max_point[2] - min_point[2]

        # Calculate the volume
        volume = length_x * length_y * length_z

        return volume
    

    def overlap_volume(self, bounding_box1, bounding_box2):
        """
        Calculate the volume of the intersection of two 3D bounding boxes.

        Parameters:
        - box1_min: tuple of (x_min, y_min, z_min) for the first bounding box
        - box1_max: tuple of (x_max, y_max, z_max) for the first bounding box
        - box2_min: tuple of (x_min, y_min, z_min) for the second bounding box
        - box2_max: tuple of (x_max, y_max, z_max) for the second bounding box

        Returns:
        - The volume of the intersecting region, or 0 if there is no intersection.
        """
        box1_min = [bounding_box1[0].x, bounding_box1[0].y, bounding_box1[0].z]
        box1_max = [bounding_box1[1].x, bounding_box1[1].y, bounding_box1[1].z]
        box2_min = [bounding_box2[0].x, bounding_box2[0].y, bounding_box2[0].z]
        box2_max = [bounding_box2[1].x, bounding_box2[1].y, bounding_box2[1].z]

        # Calculate overlap in each dimension
        x_overlap = max(0, min(box1_max[0], box2_max[0]) - max(box1_min[0], box2_min[0]))
        y_overlap = max(0, min(box1_max[1], box2_max[1]) - max(box1_min[1], box2_min[1]))
        z_overlap = max(0, min(box1_max[2], box2_max[2]) - max(box1_min[2], box2_min[2]))

        # Calculate the volume of the intersecting region
        overlap_vol = x_overlap * y_overlap * z_overlap

        return overlap_vol
    
    
    def polygon_area(self, vertices):
        
        # Number of vertices
        n = len(vertices)
        
        # Initialize area
        area = 0
        
        # Calculate the Shoelace formula
        for i in range(n):
            x1 = vertices[i].x
            y1 = vertices[i].y
            x2 = vertices[(i + 1) % n].x  # Wraps around to the first vertex
            y2 = vertices[(i + 1) % n].y 
            area += x1 * y2 - y1 * x2
        
        # Return the absolute value of the result divided by 2
        return abs(area) / 2
    
    
    def has_adjacent_points(self, polygon_1, polygon_2):
        
        for i in range(0, len(polygon_1), 10):
            for j in range(0, len(polygon_2), 10):
                point_1 = (polygon_1[i].x, polygon_1[i].y, polygon_1[i].z)
                point_2 = (polygon_2[j].x, polygon_2[j].y, polygon_2[j].z)
                
                if self.euclidean_distance2d(point_1, point_2) < 0.5:
                    return True
                
        return False
    
    
if __name__ == '__main__':
    rospy.init_node('graph_management_node')
    
    try:
        graph_node = GraphManagementNode()
        graph_node.run_main_loop()
    except rospy.ROSInterruptException:
        pass
