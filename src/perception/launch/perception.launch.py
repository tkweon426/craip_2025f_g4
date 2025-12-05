#!/usr/bin/env python3
"""
Launch file for the perception module.

This launches the perception node with configurable parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='yolov8n.pt',
        description='Path to YOLO model file (.pt)'
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Confidence threshold for detections (0.0 - 1.0)'
    )

    distance_threshold_arg = DeclareLaunchArgument(
        'distance_threshold',
        default_value='3.0',
        description='Maximum distance (meters) for bark trigger'
    )

    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='/camera_top/image',
        description='RGB image topic'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera_top/depth',
        description='Depth image topic'
    )

    # Create the perception node
    perception_node = Node(
        package='perception',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'distance_threshold': LaunchConfiguration('distance_threshold'),
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
        }]
    )

    return LaunchDescription([
        model_path_arg,
        confidence_threshold_arg,
        distance_threshold_arg,
        rgb_topic_arg,
        depth_topic_arg,
        perception_node,
    ])

