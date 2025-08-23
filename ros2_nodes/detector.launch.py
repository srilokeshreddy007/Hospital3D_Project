from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="ros2run",  # not used; we run a plain python file
            executable="python",
            name="hospital3d_detector",
            namespace="",
            output="screen",
            arguments=["ros2_nodes/detector_node.py"],
            parameters=[{
                "cloud_topic": "/camera/points",
                "cloud_frame": "camera_link",
                "costmap_frame": "map",
                "ckpt_path": "outputs/best_sota_rebal.pt",
                "num_points": 1024,
                "publish_topic": "/detected_obstacles",
                "resolution": 0.05,
                "roi_xmin": 0.3,
                "roi_xmax": 3.0,
                "roi_yabs": 1.5,
                "roi_zabs": 1.5,
                "publish_markers": True,
            }],
        )
    ])
