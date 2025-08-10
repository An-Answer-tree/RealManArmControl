from dataclasses import asdict
import cv2
import math
import matplotlib.pyplot as plt
import os
import sys
from termcolor import colored


from pyorbbecsdk import OBSensorType
from realmanarmcontrol.base.rm_robot_interface import *
from realmanarmcontrol.arm.arm_controller import RobotArmController
from realmanarmcontrol.sensor.orbbec_controller import Gemini335Controller
from realmanarmcontrol.arm.config import ArmDefault
from realmanarmcontrol.sensor.config import Gemini335Config

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
        ip=ArmDefault.ip,
        level=ArmDefault.level,
        mode=ArmDefault.mode
        )
    print(colored(line, color="yellow"))
    
    # Connect Camera, Set Default and Start
    # print(colored("Camera Initialization: ", "light_red"))
    # gemini_controller = Gemini335Controller(
    #     RGB_width=Gemini335Config.RGB_width,
    #     RGB_height=Gemini335Config.RGB_height,
    #     RGB_fps=Gemini335Config.RGB_fps,
    #     Depth_width=Gemini335Config.Depth_width,
    #     Depth_height=Gemini335Config.Depth_height,
    #     Depth_fps=Gemini335Config.Depth_fps,
    # )
    # print(colored(line, color="yellow"))


    # Take photos
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

    
    # Move looking at
    eef_pos = [0.5, 0, 0]
    look_at = [0.5, 0, -0.5]
    arm_controller.movej_p_look_at(eef_pos, look_at, v=30)


    # Disconnect
    arm_controller.disconnect()
    # gemini_controller.disconnect()
    

if __name__ == "__main__":
    main()



    