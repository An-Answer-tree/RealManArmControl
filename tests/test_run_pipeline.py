import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from termcolor import colored

from realmanarmcontrol.pipeline.camera_arm_pipeline import Pipeline
from realmanarmcontrol.pipeline.pipleline_config import PipelineConfig
from realmanarmcontrol.algorithms.yolo.run_detect import Detector

# Create arm and camera connect
config = PipelineConfig()
pipeline = Pipeline(config)

# Gripper the equipment
pipeline.arm_controller.set_gripper_pos(150)
time.sleep(3)
pipeline.arm_controller.gripper_keep_pick(speed=500, force=1000, block=False)
time.sleep(5)

# Take photo
time_stamp, image, depth_img, point_cloud = pipeline.camera_controller.take_photo()

# Run YOLO Detection
Detector = Detector(config.model_path)
points = Detected_points = Detector.get_detect_points(image)

# print(points[:, 0])
# print(points[:, 1])
# plt.figure(figsize=(8, 6))
# plt.imshow(image)
# # Mark the points in red, size 100
# plt.scatter(points[:, 0], points[:, 1], c='red', s=20, marker='o')
# # plt.title(f"RGB Image with marked point ({u}, {v})")
# plt.axis('off')
# plt.show()

# Get camera points
camera_points = []
for curr_point in points:
    u = curr_point[0]
    v = curr_point[1]
    camera_point = pipeline.camera_controller.get_point_xyz_from_pointcloud(point_cloud, u, v)
    camera_point = tuple(v / 1000.0 for v in camera_point)
    camera_points.append(camera_point)
print(colored(f"Camera points: {camera_points}", ""))

for i, pt in enumerate(camera_points):
    if tuple(pt) == (0, 0, 0):
        raise ValueError(f"camera_points[{i}] is all zeros")


# Camera_to_base
# curr_xyz = pipeline.arm_controller.get_current_end_pose()[0:3]
# print(f"Current eef xyz on base: {curr_xyz}")
camera_point = camera_points[0]
# camera_point = pipeline.camera_calibration
# tool_point = pipeline.camera_point_to_tool(camera_point)
# print(f"Camera point to tool point: {tool_point}")
base_point = pipeline.camera_point_to_base(camera_point)
# print(f"Base point: {base_point}")

# camera_point = (0.068707726, 0.07986717, 0.690)
# base_point = pipeline.camera_point_to_base(camera_point)
# print(base_point)



# Move Arm
target = base_point
lookat = target - [-0.005, 0, 0.005]
print(target)
print(lookat)
pipeline.arm_controller.movej_p_look_at(target, lookat, v=5)

time.sleep(5)
pipeline.arm_controller.movej(config.default_arm_joint)

# Disconnect arm and camera
pipeline.disconnect()