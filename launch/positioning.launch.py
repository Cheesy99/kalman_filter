from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, LogInfo
import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue



def generate_launch_description():
    """
    Launch file for Kalman-based positioning node with landmark-based localization.
    
    This launch file starts the UKF positioning node that fuses:
    - Odometry data from /robot_noisy topic
    - Landmark observations from /landmarks_observed topic
    
    Output: Estimated pose and twist published to /robot_estimated_odometry
    """
    
    # Get the package directory
    pkg_dir = get_package_share_directory('kalman_positioning')
    
    # Default landmarks CSV path
    landmarks_csv_default = os.path.join(pkg_dir, 'landmarks.csv')
    
    # Launch arguments with descriptions
    landmarks_csv_arg = DeclareLaunchArgument(
        'landmarks_csv_path',
        default_value=landmarks_csv_default,
        description='Path to CSV file containing landmark positions (id,x,y format)'
    )
    
    process_noise_xy_arg = DeclareLaunchArgument(
        'process_noise_xy',
        default_value='1e-4',
        description='Process noise covariance for x,y position. Increase to trust odometry less.'
    )
    
    process_noise_theta_arg = DeclareLaunchArgument(
        'process_noise_theta',
        default_value='1e-4',
        description='Process noise covariance for orientation. Increase to trust orientation less.'
    )
    
    measurement_noise_xy_arg = DeclareLaunchArgument(
        'measurement_noise_xy',
        default_value='0.01',
        description='Measurement noise covariance for landmark observations. Increase if observations are noisy.'
    )
    
    observation_radius_arg = DeclareLaunchArgument(
        'observation_radius',
        default_value='5.0',
        description='Maximum radius (meters) for observing landmarks from robot position'
    )
    
    # Optional logging level argument
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        choices=['debug', 'info', 'warn', 'error', 'fatal'],
        description='Logging level for the positioning node'
    )
    
    # Optional namespace argument
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='ROS namespace for the node (empty string for no namespace)'
    )
    
    # Positioning Node
    positioning_node = Node(
        package='kalman_positioning',
        executable='positioning_node',
        name='positioning_node',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {
                 'landmarks_csv_path': LaunchConfiguration('landmarks_csv_path'),

        'process_noise_xy': ParameterValue(
            LaunchConfiguration('process_noise_xy'), value_type=float
        ),
        'process_noise_theta': ParameterValue(
            LaunchConfiguration('process_noise_theta'), value_type=float
        ),
        'measurement_noise_xy': ParameterValue(
            LaunchConfiguration('measurement_noise_xy'), value_type=float
        ),
        'observation_radius': ParameterValue(
            LaunchConfiguration('observation_radius'), value_type=float
        ),
            }
        ],
        remappings=[
            # Uncomment and modify if you want to remap input/output topics
            # ('/robot_noisy', '/odometry/noisy'),
            # ('/landmarks_observed', '/sensor/landmarks'),
            # ('/robot_estimated_odometry', '/odometry/filtered'),
        ]
    )
    
    # Info message about the launch
    log_info = LogInfo(
        msg=['Starting Kalman Positioning Node with parameters:',
             ' - Landmarks CSV: ', LaunchConfiguration('landmarks_csv_path'),
             ' - Process Noise XY: ', LaunchConfiguration('process_noise_xy'),
             ' - Process Noise Theta: ', LaunchConfiguration('process_noise_theta'),
             ' - Measurement Noise XY: ', LaunchConfiguration('measurement_noise_xy'),
             ' - Observation Radius: ', LaunchConfiguration('observation_radius')]
    )
    
    return LaunchDescription([
        landmarks_csv_arg,
        process_noise_xy_arg,
        process_noise_theta_arg,
        measurement_noise_xy_arg,
        observation_radius_arg,
        log_level_arg,
        namespace_arg,
        log_info,
        positioning_node,
    ])
