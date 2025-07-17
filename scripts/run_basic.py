import os
import sys
import math
from dataclasses import asdict

from realmanarmcontrol.base.rm_robot_interface import *
from realmanarmcontrol.arm.controller import RobotArmController
from realmanarmcontrol.arm.config import ArmDefault

def main():
    # Connect Robot and Set Default
    controller: RobotArmController = RobotArmController(
        default_pos=ArmDefault.default_pos,
        default_work_frame=ArmDefault.default_work_frame,
        default_tool_frame=ArmDefault.default_tool_frame,
        default_arm_joint=ArmDefault.default_arm_joint,
        ip=ArmDefault.ip,
        level=ArmDefault.level,
        mode=ArmDefault.mode
        )
    
    # Move looking at
    eef_pos = [0.5, 0, 0]
    look_at = [0.5, 0, -0.5]

    controller.movej_p_look_at(eef_pos, look_at, v=30)





    # Disconnect
    controller.disconnect()
    

    


if __name__ == "__main__":
    main()



    