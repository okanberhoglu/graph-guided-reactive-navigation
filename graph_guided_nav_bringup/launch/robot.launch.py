import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('graph_guided_nav_bringup'), 'rviz', 'robot.rviz')

    use_sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false', description='Use the simulator /clock'),
        DeclareLaunchArgument('rviz', default_value='true', description='Start RViz'),

        Node(
            package='local_map_creator',
            executable='map_creator_node',
            parameters=[{'is_car': False, 'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='global_motion_planner',
            executable='motion_planner_node',
            parameters=[{'is_car': False, 'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='robot_controller',
            executable='controller_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
