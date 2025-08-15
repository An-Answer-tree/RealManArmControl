import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from termcolor import colored

from realmanarmcontrol.pipeline.camera_arm_pipeline import Pipeline
from realmanarmcontrol.pipeline.pipleline_config import PipelineConfig
from realmanarmcontrol.algorithms.yolo.run_detect import Detector

def ultrasound_search(pipeline: Pipeline):
    """Wiggle the end-effector orientation while keeping XYZ fixed.

    This routine keeps the current TCP position unchanged and applies small
    intrinsic-XYZ Euler offsets (in radians) to “wiggle” the tool orientation
    around the current pose. Offsets are executed in the **tool frame**
    (frame_flag=1) via ``movel_offset``; translation offsets are zero.

    Args:
        pipeline: Active Pipeline instance that provides access to the arm controller.

    Returns:
        None. Prints colored status for each step.

    Notes:
        This function is intended for small angular perturbations to probe or
        refine a sensing/ultrasound pose while holding the TCP position constant.
    """
    # --- configuration ---
    delta_deg = 5.0                     # Small angular step in degrees
    v, r, connect, block = 10, 0, 1, 0  # Conservative motion parameters
    delta = np.deg2rad(delta_deg)

    # --- read current pose (for logging only) ---
    current_pose = pipeline.arm_controller.get_current_end_pose()
    pos = np.array(current_pose[:3], dtype=float)
    euler = np.array(current_pose[3:], dtype=float)
    print(colored("Current TCP pose:", "blue"),
          colored(f"pos={pos.tolist()}, euler(XYZ,rad)={euler.tolist()}", "cyan"))

    # --- build a small wobble pattern: +rx, -rx, +ry, -ry (keep xyz fixed) ---
    offsets = np.array([
        [0.0, 0.0, 0.0, +delta, 0.0,   0.0],
        [0.0, 0.0, 0.0, -delta, 0.0,   0.0],
        [0.0, 0.0, 0.0,  0.0,  +delta, 0.0],
        [0.0, 0.0, 0.0,  0.0,  -delta, 0.0],
        # To add a slight yaw (rz) wobble, uncomment the following:
        # [0.0, 0.0, 0.0,  0.0,  0.0,  +delta],
        # [0.0, 0.0, 0.0,  0.0,  0.0,  -delta],
    ], dtype=float)

    # --- execute pattern ---
    for i, off in enumerate(offsets, start=1):
        offset = current_pose + off
        status = pipeline.arm_controller.movej_p(
            pose=offset,
            v=v,
            r=r,
            connect=connect,
            block=block
        )
    print()
        

if __name__ == "__main__":
    # Create arm and camera connect
    config = PipelineConfig()
    pipeline = Pipeline(config)


    # Gripper the equipment
    # pipeline.arm_controller.set_gripper_pos(150)
    # time.sleep(3)
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
    # Check Point
    for i, pt in enumerate(camera_points):
        if tuple(pt) == (0, 0, 0):
            raise ValueError(colored(f"camera_points[{i}] is all zeros", "red"))
        
    base_points = []
    for camera_point in camera_points:
        # Camera point to base point
        base_point = pipeline.camera_point_to_base(camera_point)
        base_points.append(base_point)

    for base_point in base_points:
        # Move Arm
        target = base_point
        lookat = target - [-0.1, 0, 0.5]
        pipeline.arm_controller.movej_p_look_at(target, lookat, v=10)
        ultrasound_search(pipeline)
        # time.sleep(2)
        target = target + np.array([0, 0, 0.01])
        pipeline.arm_controller.movej_p_look_at(target, lookat, v=5)


    # Lift arm
    time.sleep(5)
    pipeline.arm_controller.movej(config.default_arm_joint)


    # Disconnect arm and camera
    pipeline.disconnect()