#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge, CvBridgeError


class YOLODetector(Node):
    """
    Skeleton YOLO detector node for object detection.
    This is Phase 1 - basic structure with RGB subscription.
    Will be expanded in Phase 4 with full YOLO inference.
    """

    def __init__(self):
        super().__init__('yolo_detector')

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Publishers (required topics)
        self.image_pub = self.create_publisher(
            Image, '/camera/detections/image', 10)
        self.labels_pub = self.create_publisher(
            String, '/detections/labels', 10)
        self.distance_pub = self.create_publisher(
            Float32, '/detections/distance', 10)

        # Subscriber to RGB camera
        self.rgb_sub = self.create_subscription(
            Image, '/camera_top/image', self.rgb_callback, 10)

        # Frame counter for logging
        self.frame_count = 0

        self.get_logger().info('YOLO Detector initialized (skeleton mode)')
        self.get_logger().info('Subscribed to: /camera_top/image')
        self.get_logger().info('Publishing to: /camera/detections/image, /detections/labels, /detections/distance')

    def rgb_callback(self, msg):
        """Process RGB images - skeleton version"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            self.frame_count += 1

            # Log periodically (every 30 frames)
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f'Received frame {self.frame_count}, '
                    f'size: {cv_image.shape[1]}x{cv_image.shape[0]}'
                )

            # Publish skeleton outputs
            # In Phase 4, this will contain actual detections

            # 1. Publish the same image back (no annotations yet)
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)

            # 2. Publish empty labels
            labels_msg = String()
            labels_msg.data = 'None'
            self.labels_pub.publish(labels_msg)

            # 3. Publish invalid distance
            distance_msg = Float32()
            distance_msg.data = -1.0
            self.distance_pub.publish(distance_msg)

        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error in rgb_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Shutting down. Processed {node.frame_count} frames.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
