from dataclasses import asdict
import cv2
import json
import math
import matplotlib.pyplot as plt
import os
from pprint import pprint
import sys
from termcolor import colored
import time


from pyorbbecsdk import OBSensorType
from realmanarmcontrol.base.rm_robot_interface import *
from realmanarmcontrol.controller.arm.arm_controller import RobotArmController
from realmanarmcontrol.controller.sensor.orbbec_controller import Gemini335Controller
from realmanarmcontrol.controller.arm.arm_config import ArmDefault
from realmanarmcontrol.controller.sensor.camera_config import Gemini335Config

def main():
    line = '*' * 150
    print(colored(line, color="yellow"))

    # Connect Robot and Set Default
    print(colored("Arm Initialization: ", "light_red"))
    arm_controller = RobotArmController(
        set_default_pos=ArmDefault.set_default_pos,
        default_work_frame=ArmDefault.default_work_frame,
        default_tool_frame=ArmDefault.default_tool_frame,
        default_arm_joint=ArmDefault.default_arm_joint,
        have_gripper=ArmDefault.have_gripper,
        ip=ArmDefault.ip,
        level=ArmDefault.level,
        mode=ArmDefault.mode
        )
    print(colored(line, color="yellow"))
    
    # Connect Camera, Set Default and Start
    print(colored("Camera Initialization: ", "light_red"))
    gemini_controller = Gemini335Controller(
        RGB_width=Gemini335Config.RGB_width,
        RGB_height=Gemini335Config.RGB_height,
        RGB_fps=Gemini335Config.RGB_fps,
        Depth_width=Gemini335Config.Depth_width,
        Depth_height=Gemini335Config.Depth_height,
        Depth_fps=Gemini335Config.Depth_fps,
    )
    print(colored(line, color="yellow"))

    # ======================Algo tool2base======================
    # arm_controller.movej_p_look_at([0.5, 0, 0.6], [0.6, 0, 0.6], up=[0, 0, 1], v=5)
    # arm_controller.get_current_end_pose()[0:3]
    # point_base = arm_controller.tool_point_to_base((0.3, 0.2, 0))
    # print(point_base)

    # ======================Tool Frame Setting test======================
    # tool_frames = arm_controller.get_all_tool_frames()
    # print(tool_frames)
    # arm_controller.change_tool_frame("Arm_Tip")

    # ======================Gripper test======================
    # arm_controller.release_gripper(speed=500)
    # arm_controller.gripper_keep_pick(speed=50, force=1000, timeout=100)
    # while True:
    #     gripper_status = arm_controller.get_gripper_status()
    #     print(gripper_status)
    # arm_controller.set_gripper_pos(150)
    # time.sleep(5)
    arm_controller.gripper_keep_pick(speed=500, force=1000, block=True)

    # ======================Record Joint Qpos======================
    # joint_list = list()
    # for i in range(30):
    #     input("press enter")
    #     qpos = arm_controller.get_current_joint_angles()
    #     joint_list.append(qpos)
    #     pprint(joint_list)

    # ======================Record dataset======================
    # joint_list = [[0.0, 
    #                44.99800109863281, 
    #                -100.0, -0.0010000000474974513, 
    #                -90.0, 
    #                0.0],
    #                 [-0.03200000151991844,
    #                 6.181000232696533,
    #                 -33.85300064086914,
    #                 -3.993000030517578,
    #                 -114.01399993896484,
    #                 0.0020000000949949026]]
    
    # for joint in joint_list:
    #     arm_controller.movej(joint, v=50)
    #     time.sleep(0.5)
        
    #     # Take Photo and save status
    #     time_stamp, RGB_image, Depth_image, Points = gemini_controller.take_photo()

    #     # Save joint pos
    #     time_stamp = gemini_controller.time_stamp
    #     save_dir = Gemini335Config.saved_path
    #     # Create save path
    #     save_dir_pose = os.path.join(save_dir, "arm_pose")
    #     os.makedirs(save_dir_pose, exist_ok=True)
    #     # Get status
    #     current_joints = arm_controller.get_current_joint_angles()
    #     current_eef = arm_controller.get_current_end_pose()
    #     current_pose = {"joints": current_joints, "eef": current_eef}
    #     # Save Pose json
    #     basename = f"JointPose_{time_stamp}.json"
    #     filepath = os.path.join(save_dir_pose, basename)
    #     with open(filepath, "w", encoding="utf-8") as f:
    #         json.dump(current_pose, f, ensure_ascii=False, indent=4)

    # ======================Move======================
    # arm_controller.movej([0.0, 45.0, -100.0, 0.0, -90.0, 0.0])

    # ======================Take photos======================
    # time_stamp, RGB_image, Depth_image, Points = gemini_controller.take_photo()
    # u, v = (400, 200)
    # u2, v2 = (880, 200)
    # rgb_plot = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)

    # print(gemini_controller.get_point_xyz_from_pointcloud(Points, u, v))
    # print(gemini_controller.get_point_xyz_from_pointcloud(Points, u2, v2))
    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(rgb_plot)
    # # Mark the point (u, v) in red, size 100
    # plt.scatter([u,u2], [v,v2], c='red', s=100, marker='o')
    # plt.title(f"RGB Image with marked point ({u}, {v})")
    # plt.axis('off')
    # plt.show()

    
    # ======================Move looking at======================
    # eef_pos = [0.7, 0, 0.7]
    # look_at = [0.8, 0, 0.7]
    # arm_controller.movej_p_look_at(eef_pos, look_at, v=5, up=[0, 0, 1])


    # ======================Disconnect======================
    arm_controller.disconnect()
    gemini_controller.disconnect()
    

if __name__ == "__main__":
    main()



    