from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'save_dir',
            default_value='/home/tkweon426/craip_2025f_g4/src/perception/data/images/train',
            description='Directory to save collected images'
        ),
        DeclareLaunchArgument(
            'capture_interval',
            default_value='2.0',
            description='Interval between image captures in seconds'
        ),

        Node(
            package='perception',
            executable='data_collector',
            name='data_collector',
            output='screen',
            parameters=[{
                'save_dir': LaunchConfiguration('save_dir'),
                'capture_interval': LaunchConfiguration('capture_interval'),
            }]
        )
    ])
