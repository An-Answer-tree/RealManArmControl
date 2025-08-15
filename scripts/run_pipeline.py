import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from realmanarmcontrol.pipeline.camera_arm_pipeline import Pipeline
from realmanarmcontrol.pipeline.pipleline_config import PipelineConfig
from realmanarmcontrol.algorithms.yolo.run_detect import Detector

# Create arm and camera connect
config = PipelineConfig()
pipeline = Pipeline(config)

# Take photo
# time_stamp, image, depth_img, point_cloud = pipeline.camera_controller.take_photo()

# Run YOLO Detection
# Detector = Detector(config.model_path)
# points = Detected_points = Detector.get_detect_points(image)

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
# camera_points = []
# for curr_point in points:
#     u = curr_point[0]
#     v = curr_point[1]
#     camera_point = pipeline.camera_controller.get_point_xyz_from_pointcloud(point_cloud, u, v)
#     camera_points.append(camera_point)
# print(f"Camera points: {camera_points}")

# Camera_to_base
# curr_xyz = pipeline.arm_controller.get_current_end_pose()[0:3]
# print(f"Current eef xyz on base: {curr_xyz}")
# # camera_point = camera_points[0]
# camera_point = pipeline.camera_calibration
# tool_point = pipeline.camera_point_to_tool(camera_point)
# print(f"Camera point to tool point: {tool_point}")
# base_point = pipeline.arm_controller.tool_point_to_base(tool_point)
# print(f"Base point: {base_point}")
camera_point = (0.068707726, 0.07986717, 0.690)
base_point = pipeline.camera_point_to_base(camera_point)
print(base_point)



# Move Arm
target = base_point
lookat = target - [0, 0, 0.5]
print(lookat)
pipeline.arm_controller.movej_p_look_at(target, lookat, v=5)

# Disconnect arm and camera
pipeline.camera_controller.reboot()
pipeline.disconnect()