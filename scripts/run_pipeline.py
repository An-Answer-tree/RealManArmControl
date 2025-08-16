import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from termcolor import colored

from realmanarmcontrol.pipeline.camera_arm_pipeline import Pipeline
from realmanarmcontrol.pipeline.pipleline_config import PipelineConfig
from realmanarmcontrol.algorithms.yolo.run_detect import Detector
        

if __name__ == "__main__":
    # Create arm and camera connect
    config = PipelineConfig()
    pipeline = Pipeline(config)


    # Gripper the equipment
    # pipeline.arm_controller.set_gripper_pos(150)
    # time.sleep(5)
    pipeline.arm_controller.gripper_keep_pick(speed=500, force=1000, block=False)
    # time.sleep(5)


    # Take photo
    time_stamp, image, depth_img, point_cloud = pipeline.camera_controller.take_photo()


    # Run YOLO Detection
    Detector = Detector(config.model_path)
    points = Detected_points = Detector.get_detect_points(image)


    # Get camera points
    camera_points = []
    for curr_point in points:
        u = curr_point[0]
        v = curr_point[1]
        camera_point = pipeline.camera_controller.get_point_xyz_from_pointcloud(point_cloud, u, v)
        camera_point = tuple(v / 1000.0 for v in camera_point)
        camera_points.append(camera_point)
    print(colored(f"Camera points: {camera_points}\n", "cyan"))
    # Check camera points
    for i, pt in enumerate(camera_points):
        if tuple(pt) == (0, 0, 0):
            raise ValueError(colored(f"camera_points[{i}] is all zeros", "red"))
    
    # Convert camera points to base points
    base_points = []
    for camera_point in camera_points:
        # Camera point to base point
        base_point = pipeline.camera_point_to_base(camera_point)
        base_points.append(base_point)
    print(colored(f"Camera to Base points: {base_points}\n", "cyan"))


    # Execute ultra sound detection
    count = 1
    for base_point in base_points:
        # Move Arm
        target = base_point
        lookat = target - [-0.1, 0, 0.4]
        pipeline.arm_controller.movej_p_look_at(target, lookat, v=10)
        if count == 2:
            for _ in range(3):
                pipeline.ultrasound_search()
        else:
            pipeline.ultrasound_search(v=5, r=0, connect=0, block=1)
        target = target + np.array([0, 0, 0.01])
        pipeline.arm_controller.movej_p_look_at(target, lookat, v=5)
        count += 1


    # Lift arm
    time.sleep(1)
    pipeline.arm_controller.movej(config.default_arm_joint)


    # Disconnect arm and camera
    pipeline.disconnect()