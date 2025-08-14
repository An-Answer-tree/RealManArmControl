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

    # joint_list = list()
    # for i in range(30):
    #     input("press enter")
    #     qpos = arm_controller.get_current_joint_angles()
    #     joint_list.append(qpos)
    #     pprint(joint_list)


    joint_list = [[0.0, 
                   44.99800109863281, 
                   -100.0, -0.0010000000474974513, 
                   -90.0, 
                   0.0],
                    [-0.03200000151991844,
                    6.181000232696533,
                    -33.85300064086914,
                    -3.993000030517578,
                    -114.01399993896484,
                    0.0020000000949949026],
                    [-0.020999999716877937,
                    27.820999145507812,
                    -67.46299743652344,
                    0.984000027179718,
                    -106.63400268554688,
                    0.0020000000949949026],
                    [-0.01600000075995922,
                    23.731000900268555,
                    -85.30699920654297,
                    -5.28000020980835,
                    -77.44100189208984,
                    0.004999999888241291],
                    [-0.029999999329447746,
                    43.29399871826172,
                    -103.49800109863281,
                    -5.291999816894531,
                    -78.18000030517578,
                    0.019999999552965164],
                    [-0.01899999938905239,
                    58.51300048828125,
                    -111.14399719238281,
                    -5.283999919891357,
                    -78.26899719238281,
                    0.014999999664723873],
                    [-0.014999999664723873,
                    63.28300094604492,
                    -122.91500091552734,
                    -5.2729997634887695,
                    -67.76300048828125,
                    0.014999999664723873],
                    [-0.012000000104308128,
                    44.77000045776367,
                    -122.927001953125,
                    -5.264999866485596,
                    -51.720001220703125,
                    0.013000000268220901],
                    [-0.008999999612569809,
                    31.054000854492188,
                    -122.9229965209961,
                    -5.26200008392334,
                    -45.06999969482422,
                    0.019999999552965164],
                    [-0.006000000052154064,
                    19.02199935913086,
                    -118.21399688720703,
                    -5.267000198364258,
                    -39.24700164794922,
                    0.01899999938905239],
                    [-0.03099999949336052,
                    18.23699951171875,
                    -114.43599700927734,
                    -5.269000053405762,
                    -51.22700119018555,
                    0.006000000052154064],
                    [-0.03200000151991844,
                    10.48900032043457,
                    -103.19300079345703,
                    -5.269999980926514,
                    -48.8390007019043,
                    0.013000000268220901],
                    [-0.06499999761581421,
                    10.46399974822998,
                    -99.01100158691406,
                    0.4880000054836273,
                    -66.08599853515625,
                    0.02199999988079071],
                    [-0.04899999871850014,
                    0.9369999766349792,
                    -93.1719970703125,
                    0.4830000102519989,
                    -62.22100067138672,
                    0.03099999949336052],
                    [-0.05999999865889549,
                    31.434999465942383,
                    -93.21700286865234,
                    0.4880000054836273,
                    -87.53900146484375,
                    0.02800000086426735],
                    [-0.06300000101327896,
                    31.52199935913086,
                    -82.76799774169922,
                    0.4830000102519989,
                    -87.58100128173828,
                    0.027000000700354576],
                    [-0.07000000029802322,
                    30.577999114990234,
                    -74.5250015258789,
                    -6.413000106811523,
                    -102.2300033569336,
                    0.023000000044703484],
                    [-0.07400000095367432,
                    40.79399871826172,
                    -74.93699645996094,
                    -6.40500020980835,
                    -98.53500366210938,
                    0.019999999552965164],
                    [-0.06499999761581421,
                    26.139999389648438,
                    -74.06099700927734,
                    -6.392000198364258,
                    -98.4990005493164,
                    0.019999999552965164],
                    [-0.05999999865889549,
                    15.25100040435791,
                    -59.08100128173828,
                    2.1610000133514404,
                    -105.66899871826172,
                    0.02199999988079071],
                    [-0.10899999737739563,
                    31.764999389648438,
                    -84.24400329589844,
                    1.3559999465942383,
                    -97.33300018310547,
                    0.17800000309944153],
                    [-0.11999999731779099,
                    39.555999755859375,
                    -96.84700012207031,
                    1.2289999723434448,
                    -83.13200378417969,
                    0.1899999976158142],
                    [-0.1289999932050705,
                    30.742000579833984,
                    -86.93199920654297,
                    1.218000054359436,
                    -94.78299713134766,
                    0.1860000044107437],
                    [-0.12700000405311584,
                    22.98699951171875,
                    -70.30899810791016,
                    1.218000054359436,
                    -108.53299713134766,
                    0.18199999630451202],
                    [-0.1340000033378601,
                    36.551998138427734,
                    -70.33000183105469,
                    1.2230000495910645,
                    -97.6729965209961,
                    0.17900000512599945],
                    [-0.14100000262260437,
                    46.138999938964844,
                    -70.40699768066406,
                    0.953000009059906,
                    -108.44400024414062,
                    0.1889999955892563],
                    [-1.3650000095367432,
                    45.854000091552734,
                    -121.36699676513672,
                    -1.0160000324249268,
                    -60.01100158691406,
                    0.21400000154972076],
                    [-1.3559999465942383,
                    45.555999755859375,
                    -109.01599884033203,
                    -1.0140000581741333,
                    -74.87799835205078,
                    -21.479999542236328],
                    [-1.3550000190734863,
                    33.5629997253418,
                    -91.9990005493164,
                    -6.877999782562256,
                    -93.74299621582031,
                    -17.756999969482422]]
    
    for joint in joint_list:
        arm_controller.movej(joint, v=50)
        time.sleep(0.5)
        
        # Take Photo and save status
        time_stamp, RGB_image, Depth_image, Points = gemini_controller.take_photo()

        # Save joint pos
        time_stamp = gemini_controller.time_stamp
        save_dir = Gemini335Config.saved_path
        # Create save path
        save_dir_pose = os.path.join(save_dir, "arm_pose")
        os.makedirs(save_dir_pose, exist_ok=True)
        # Get status
        current_joints = arm_controller.get_current_joint_angles()
        current_eef = arm_controller.get_current_end_pose()
        current_pose = {"joints": current_joints, "eef": current_eef}
        # Save Pose json
        basename = f"JointPose_{time_stamp}.json"
        filepath = os.path.join(save_dir_pose, basename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(current_pose, f, ensure_ascii=False, indent=4)

    # Move
    # arm_controller.movej([0, 0, 0, 0, 0, 0])

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
    # eef_pos = [0.5, 0, 0]
    # look_at = [0.5, 0, -0.5]
    # arm_controller.movej_p_look_at(eef_pos, look_at, v=30)

    # # Save joint pos
    # time_stamp = gemini_controller.time_stamp
    # save_dir = Gemini335Config.saved_path
    # # Create save path
    # save_dir_pose = os.path.join(save_dir, "arm_pose")
    # os.makedirs(save_dir_pose, exist_ok=True)
    # # Get status
    # current_joints = arm_controller.get_current_joint_angles()
    # current_eef = arm_controller.get_current_end_pose()
    # current_pose = {"joints": current_joints, "eef": current_eef}
    # # Save Pose json
    # basename = f"JointPose_{time_stamp}.json"
    # filepath = os.path.join(save_dir_pose, basename)
    # with open(filepath, "w", encoding="utf-8") as f:
    #     json.dump(current_pose, f, ensure_ascii=False, indent=4)



    # Disconnect
    arm_controller.disconnect()
    gemini_controller.disconnect()
    

if __name__ == "__main__":
    main()



    