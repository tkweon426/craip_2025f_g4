#!/usr/bin/env python3
"""
Perception Node for ROS2

This node performs object detection using YOLO on RGB images from the robot's camera,
estimates distance using depth images, and publishes detection results.

Subscribed Topics:
    /camera_top/image (sensor_msgs/Image): RGB image from top camera
    /camera_top/depth (sensor_msgs/Image): Depth image from top camera

Published Topics:
    /camera/detections/image (sensor_msgs/Image): RGB image with bounding boxes
    /detections/labels (std_msgs/String): Detected object labels
    /detections/distance (std_msgs/Float32): Distance to detected object
    /robot_dog/speech (std_msgs/String): "bark" if edible object centered, else "None"
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from typing import Optional, Tuple, List
import threading

# Try to import YOLO from ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")


class PerceptionNode(Node):
    """ROS2 node for object detection and distance estimation."""

    # Define edible food classes (COCO dataset class names)
    # You should update this list based on your trained model
    EDIBLE_CLASSES = [
        'banana', 'apple', 'pizza', 'sandwich', 'orange', 'broccoli', 
        'carrot', 'hot dog', 'donut', 'cake',
        # Add custom classes from your trained model here:
        'apple_good', 'banana_good', 'pizza_good',
        'applegood', 'bananagood', 'pizzagood',
        'good_apple', 'good_banana', 'good_pizza',
    ]

    # Non-edible / bad food classes
    NON_EDIBLE_CLASSES = [
        'apple_bad', 'banana_bad', 'pizza_bad',
        'applebad', 'bananabad', 'pizzabad',
        'bad_apple', 'bad_banana', 'bad_pizza',
        'rotten_apple', 'rotten_banana', 'rotten_pizza',
    ]

    def __init__(self):
        super().__init__('perception_node')

        # Declare parameters
        self.declare_parameter('model_path', 'yolov8n.pt')  # Default to YOLOv8 nano
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('distance_threshold', 3.0)  # Max distance for bark (meters)
        self.declare_parameter('rgb_topic', '/camera_top/image')
        self.declare_parameter('depth_topic', '/camera_top/depth')

        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Thread lock for synchronized access
        self.lock = threading.Lock()

        # Latest images
        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # Load YOLO model
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.get_logger().info(f"Loading YOLO model: {model_path}")
                self.model = YOLO(model_path)
                self.get_logger().info("YOLO model loaded successfully!")
            except Exception as e:
                self.get_logger().error(f"Failed to load YOLO model: {e}")
        else:
            self.get_logger().warn("YOLO not available. Install with: pip install ultralytics")

        # Create subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )

        # Create publishers
        self.detection_image_pub = self.create_publisher(Image, '/camera/detections/image', 10)
        self.labels_pub = self.create_publisher(String, '/detections/labels', 10)
        self.distance_pub = self.create_publisher(Float32, '/detections/distance', 10)
        self.speech_pub = self.create_publisher(String, '/robot_dog/speech', 10)

        # Create timer for processing (10 Hz)
        self.timer = self.create_timer(0.1, self.process_frame)

        self.get_logger().info("Perception node initialized!")
        self.get_logger().info(f"  - RGB topic: {rgb_topic}")
        self.get_logger().info(f"  - Depth topic: {depth_topic}")
        self.get_logger().info(f"  - Confidence threshold: {self.confidence_threshold}")
        self.get_logger().info(f"  - Distance threshold: {self.distance_threshold}m")

    def rgb_callback(self, msg: Image) -> None:
        """Callback for RGB image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.latest_rgb = cv_image
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")

    def depth_callback(self, msg: Image) -> None:
        """Callback for depth image."""
        try:
            # Depth images can be 16UC1 (mm) or 32FC1 (meters)
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                # Convert mm to meters
                depth_image = depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            else:
                # Try passthrough
                depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if depth_image.dtype == np.uint16:
                    depth_image = depth_image.astype(np.float32) / 1000.0

            with self.lock:
                self.latest_depth = depth_image
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def is_edible(self, class_name: str) -> bool:
        """Check if the detected class is edible food."""
        class_lower = class_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check if it's in non-edible list first
        for non_edible in self.NON_EDIBLE_CLASSES:
            if non_edible.lower() in class_lower or class_lower in non_edible.lower():
                return False
        
        # Check if it's in edible list
        for edible in self.EDIBLE_CLASSES:
            if edible.lower() in class_lower or class_lower in edible.lower():
                return True
        
        return False

    def is_centered(self, bbox_center_x: float, image_width: int) -> bool:
        """
        Check if bounding box center is within the middle 3/5 of the image.
        Excludes leftmost and rightmost 1/5 regions.
        """
        left_boundary = image_width * 0.2  # 1/5 from left
        right_boundary = image_width * 0.8  # 1/5 from right
        return left_boundary <= bbox_center_x <= right_boundary

    def get_distance_at_bbox(self, depth_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Get the distance to an object using its bounding box.
        Returns the median distance within the center region of the bbox.
        """
        x1, y1, x2, y2 = bbox
        
        # Get center region of bounding box (inner 50%)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = (x2 - x1) // 4
        h = (y2 - y1) // 4
        
        # Extract ROI
        roi_x1 = max(0, cx - w)
        roi_x2 = min(depth_image.shape[1], cx + w)
        roi_y1 = max(0, cy - h)
        roi_y2 = min(depth_image.shape[0], cy + h)
        
        roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # Filter out invalid depth values (0, inf, nan)
        valid_depths = roi[(roi > 0.1) & (roi < 100.0) & np.isfinite(roi)]
        
        if len(valid_depths) > 0:
            return float(np.median(valid_depths))
        else:
            return float('inf')

    def process_frame(self) -> None:
        """Process the latest frame and publish results."""
        with self.lock:
            rgb_image = self.latest_rgb.copy() if self.latest_rgb is not None else None
            depth_image = self.latest_depth.copy() if self.latest_depth is not None else None

        if rgb_image is None:
            return

        # Initialize results
        detected_labels: List[str] = []
        closest_distance = float('inf')
        should_bark = False
        annotated_image = rgb_image.copy()
        image_height, image_width = rgb_image.shape[:2]

        # Run YOLO detection if model is available
        if self.model is not None:
            try:
                results = self.model(rgb_image, conf=self.confidence_threshold, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        detected_labels.append(class_name)
                        
                        # Calculate bbox center
                        bbox_center_x = (x1 + x2) / 2
                        bbox_center_y = (y1 + y2) / 2
                        
                        # Get distance if depth image is available
                        distance = float('inf')
                        if depth_image is not None:
                            # Resize depth to match RGB if needed
                            if depth_image.shape[:2] != rgb_image.shape[:2]:
                                depth_resized = cv2.resize(
                                    depth_image, 
                                    (image_width, image_height),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            else:
                                depth_resized = depth_image
                            
                            distance = self.get_distance_at_bbox(depth_resized, (x1, y1, x2, y2))
                        
                        # Update closest distance
                        if distance < closest_distance:
                            closest_distance = distance
                        
                        # Check if should bark
                        is_edible_obj = self.is_edible(class_name)
                        is_centered_obj = self.is_centered(bbox_center_x, image_width)
                        is_close_enough = distance <= self.distance_threshold
                        
                        if is_edible_obj and is_centered_obj and is_close_enough:
                            should_bark = True
                        
                        # Draw bounding box
                        color = (0, 255, 0) if is_edible_obj else (0, 0, 255)  # Green for edible, red for not
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with distance
                        label = f"{class_name}: {confidence:.2f}"
                        if distance != float('inf'):
                            label += f" ({distance:.2f}m)"
                        
                        # Background for text
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            annotated_image, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width, y1), 
                            color, 
                            -1
                        )
                        cv2.putText(
                            annotated_image, 
                            label, 
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            2
                        )

            except Exception as e:
                self.get_logger().error(f"YOLO detection error: {e}")

        # Draw center region lines (for visualization)
        left_line = int(image_width * 0.2)
        right_line = int(image_width * 0.8)
        cv2.line(annotated_image, (left_line, 0), (left_line, image_height), (255, 255, 0), 1)
        cv2.line(annotated_image, (right_line, 0), (right_line, image_height), (255, 255, 0), 1)

        # Publish results
        # 1. Detection image
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            self.detection_image_pub.publish(detection_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish detection image: {e}")

        # 2. Labels
        labels_msg = String()
        labels_msg.data = ', '.join(detected_labels) if detected_labels else ''
        self.labels_pub.publish(labels_msg)

        # 3. Distance
        distance_msg = Float32()
        distance_msg.data = closest_distance if closest_distance != float('inf') else -1.0
        self.distance_pub.publish(distance_msg)

        # 4. Speech (bark)
        speech_msg = String()
        speech_msg.data = 'bark' if should_bark else 'None'
        self.speech_pub.publish(speech_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

