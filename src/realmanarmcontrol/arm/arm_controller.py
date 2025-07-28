import math
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import time
from termcolor import colored

from realmanarmcontrol.arm.config import ArmDefault
from realmanarmcontrol.base.rm_robot_interface import *
from realmanarmcontrol.base.rm_ctypes_wrap import *


class RobotArmController:
    def __init__(
            self, 
            set_default_pos: bool = ArmDefault.set_default_pos,
            default_work_frame: dict = ArmDefault.default_work_frame,
            default_tool_frame: dict = ArmDefault.default_tool_frame,
            default_arm_joint: list = ArmDefault.default_arm_joint,
            ip: str = ArmDefault.ip, 
            port: int = ArmDefault.port, 
            level: int = ArmDefault.level, 
            mode: int = ArmDefault.mode,
        ):
        """
        Initialize and connect to the robotic arm.

        Args:
            set_default_pos: Whether to set arm to default position when start.
            default_work_frame: Work frame of the robot
            default_tool_frame: Tool frame of the robot
            default_arm_joint: Default joint position
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """

        # Create Connection
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)
        
        # Set Default Work Frame
        # Get Available Work Frames
        work_frames_result = self.robot.rm_get_total_work_frame()
        return_code = work_frames_result["return_code"]
        available_work_frames = work_frames_result["work_names"]

        if return_code != 0:
            raise RuntimeError(f"Get work frames failed, error code: {tag}")
        else:
            print(colored("\nAvailable Work Frames: ", "cyan"), available_work_frames)

        # Judge Wheter Work Frame Exits
        # If exits, update the original one.
        if default_work_frame["name"] in available_work_frames:
            tag = self.robot.rm_update_work_frame(**default_work_frame)
            if tag == 0:
                self.robot.rm_change_work_frame(default_work_frame["name"])
                print(colored("✓ Update work frame succeeded", "green"), "\n", default_work_frame)
            else:
                raise RuntimeError(f"Update work frame failed, error code: {tag}")
        # If not exits, create a new work frame.
        else:
            tag = self.robot.rm_set_manual_work_frame(**default_work_frame)
            if tag == 0:
                self.robot.rm_change_work_frame(default_work_frame["name"])
                print(colored("✓ Add manual work frame succeeded", "green"), "\n", default_work_frame)
            else:
                raise RuntimeError(f"Add manual work frame failed, error code: {tag}")

        # Set Default Tool Frame
        # Get Available Tool Frames
        tool_frames_result = self.robot.rm_get_total_tool_frame()
        return_code = tool_frames_result["return_code"]
        available_tool_frames = tool_frames_result["tool_names"]

        if return_code != 0:
            raise RuntimeError(f"Get tool frames failed, error code: {tag}")
        else:
            print(colored("\nAvailable Tool Frames: ", "cyan"), available_tool_frames)        

        # Judge Wheter Tool Frame Exits
        # If exits, update the original one.
        if default_tool_frame["name"] in available_tool_frames:
            tag = self.robot.rm_update_tool_frame(**default_tool_frame)
            if tag == 0:
                self.robot.rm_change_tool_frame(default_tool_frame["name"])
                print(colored("✓ Update tool frame succeeded", "green"), "\n", default_tool_frame)
            else:
                raise RuntimeError(f"Update tool frame failed, error code: {tag}")
        # If not exits, create a new tool frame.
        else:
            tag = self.robot.rm_set_manual_tool_frame(**default_tool_frame)
            if tag == 0:
                self.robot.rm_change_tool_frame(default_tool_frame["name"])
                print(colored("✓ Add manual tool frame succeeded", "green"), "\n", default_tool_frame)
            else:
                raise RuntimeError(f"Add manual tool frame failed, error code: {tag}")

        # Set default joint position
        if set_default_pos == True:
            tag = self.robot.rm_movej(joint=default_arm_joint, v=30, r=0, connect=0, block=1)

            ret, joints = self.robot.rm_get_joint_degree()
            if ret != 0:
                raise RuntimeError(f"rm_get_joint_degree failed, code={ret}")

            joints = [f"{i:.3f}" for i in joints]
            print(colored("\n✓ Set Joint Default", "green"), colored("\nCurrent Joint Poisiton: ", "cyan"), f"{joints}")

        # Show current status
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed (code={ret})")
        pose = state['pose']

        print(colored("Current End Effector Poisiton: ", "cyan"), pose)


    def disconnect(self):
        """
        Disconnect from the robot arm.

        Returns:
            None
        """
        handle = self.robot.rm_delete_robot_arm()

        if handle == 0:
            print(colored("\n!!!Successfully disconnected from the robot arm!!!", "red"))
        else:
            raise RuntimeError("\n!!!Failed to disconnect from the robot arm!!!")


    # ----------Tool_Frames_Function----------
    def get_all_tool_frames(self):
        """
        List names of all tool frames from robot's storage

        Args:
            None

        Returns:
            tool_frames (list): List of all the tool frames' name
        """
        info = self.robot.rm_get_total_tool_frame()

        if info['return_code'] == 0:
            print("\nAvailable Tool Frames:", info['tool_names'])
        else:
            print("\nget_all_tool_frames ERR", info['return_code'])

        tool_frames = info['tool_names']
        return tool_frames

    def del_tool_frame(self, name):
        """
        Delete A tool frame from robot's storage

        Args:
            name (string): Name of the tool frame you want to delete
    
        Returns:
            None
        """
        info = self.robot.rm_delete_tool_frame(name)

        if info == 0:
            print(f"\nDelete tool frame '{name}' succeeded")
        else:
            print("\nDelete tool frame failed, error code: ", info, "")

    def change_tool_frame(self, name):
        """
        Change current tool frame

        Args:
            name (str): Name of the tool frame you want to use
        """
        info = self.robot.rm_change_tool_frame(name)

        if info == 0:
            print(f"\nChange current tool frame to '{name}' succeeded")
        else:
            print("\nChange current tool frame failed, error code: ", info, "")


    # ----------Arm_Status_Function----------
    def get_current_joint_angles(self) -> list[float]:
        """
        Query and return all current joint angles of the robot.

        Returns:
            A list of joint angles (degrees).

        Raises:
            RuntimeError: if the underlying interface call fails.
        """
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed (code={ret})")
        joints = state['joint']
        print("Current joint angles:", joints)
        return joints

    def get_current_end_pose(self) -> list[float]:
        """
        Query and return the current end-effector pose in the world frame.

        Returns:
            A 6-element list [x, y, z, rx, ry, rz], where
            x,y,z are meters and rx,ry,rz are roll/pitch/yaw in radians.

        Raises:
            RuntimeError: if the underlying interface call fails.
        """
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            raise RuntimeError(f"rm_get_current_arm_state failed (code={ret})")
        pose = state['pose']
        print("Current end-effector pose:", pose)
        return pose


    # ----------Algo_Function----------
    def get_euler_towards_object(self, eef_pos: list, target_pos: list, up: list = [0, 0, 1]):
        """
        Get [rx, ry, rz] Euler angles (in radians) for the end effector such that its Z-axis points toward a given target.
        The X-axis is always aligned as close as possible to the global Z-axis to keep the tool horizontal.

        Args:
            eef_pos (list): Current end effector 3D position [x, y, z].
            target_pos (list): 3D position of the target to look at [x, y, z].

        Returns:
            euler (list): [rx, ry, rz] in radians.
            rotation_matrix (np.ndarray): 3x3 rotation matrix representing the tool orientation.
        """
        if eef_pos == target_pos:
            raise ValueError("The eef_pose can't equal to the target_pos")

        z_axis = np.array(target_pos) - np.array(eef_pos)
        z_axis /= np.linalg.norm(z_axis)

        up_vector = np.array(up)
        project = np.dot(up_vector, z_axis) * z_axis

        x_axis = up_vector - project

        if np.linalg.norm(x_axis) < 1e-6:
            # up and z_axis are colinear
            y_guess = [0, -1, 0]
            y_guess = np.array(y_guess)
            proj = np.dot(y_guess, z_axis) * z_axis
            y_axis = y_guess - proj
            x_axis = np.cross(y_axis, z_axis)
        else:
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=1)

        r = R.from_matrix(rotation_matrix)
        return r.as_euler('xyz', degrees=False), rotation_matrix  # Return in radians

    # ----------Arm_Moving_Function----------
    def movej_p_look_at(self, target_pos: list, look_at_pos: list, v=10, r=0, connect=0, block=1):
        """
        Move the robot to a target position while maintaining its Z-axis pointing toward a specific target point.

        Args:
            target_pos (list[float]): The target 3D position [x, y, z] in meters.
            look_at_pos (list[float]): The position the Z-axis should point toward.
            v (int): Speed percentage (1–100). Default is 30.
            r (int): Blend radius percentage (0–100). 0 means no blending. Default is 0.
            connect (int): Trajectory connection flag
                - 0: Plan and execute the trajectory immediately. Do not connect with subsequent trajectories.
                - 1: Plan the current trajectory together with the next one, but do not execute immediately.
                    In blocking mode, the function will return immediately after sending the command, even if it is successfully queued.
            block (int): Blocking mode:
                - In multi-threaded mode:
                    - 0: non-blocking (returns immediately).
                    - 1: blocking (waits until motion is complete).
                - In single-threaded mode:
                    - 0: non-blocking.
                    - >0: blocking with timeout (seconds).
        """
        rx, ry, rz = self.get_euler_towards_object(eef_pos=target_pos, target_pos=look_at_pos)[0]
        pose = target_pos + [float(rx), float(ry), float(rz)]  # Combine position and orientation into 6D pose

        # Execute the motion using joint-space planning
        result = self.robot.rm_movej_p(pose, v=v, r=r, connect=connect, block=block)

        if result == 0:
            print(colored("✓ MoveJ command executed successfully.", color="green"))
            print(colored("Target pose is: ", "cyan"), colored(pose, "cyan"))
        else:
            print(f"Move command failed with return code: {result}")
        