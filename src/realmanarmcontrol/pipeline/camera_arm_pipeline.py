import numpy as np
import os
import sys
from termcolor import colored
from typing import Sequence

from realmanarmcontrol.pipeline.pipleline_config import PipelineConfig
from realmanarmcontrol.controller.arm.arm_controller import RobotArmController
from realmanarmcontrol.controller.sensor.orbbec_controller import Gemini335Controller


class Pipeline():
    """Orchestrates the robot arm and camera controllers for an eyes-on-arm pipeline.

    This class wires together configuration, device instantiation, and basic
    lifecycle operations. It reads arm and camera parameters from a
    ``PipelineConfig`` object, initializes device controllers, and exposes
    helpers to create and disconnect them.

    Attributes:
      line (str): Visual divider used for console output.
      set_default_pos (bool): Whether to move the arm to a default pose on start.
      default_work_frame (dict): Default work frame configuration for the arm.
      default_tool_frame (dict): Default tool frame configuration for the arm.
      default_arm_joint (Sequence[float]): Default joint angles for the arm.
      have_gripper (bool): Whether the arm has a gripper attached.
      ip (str): Arm controller IP address.
      port (int): Arm controller port.
      level (int): Logging level for the arm controller.
      mode (int): Threading/operation mode for the arm controller.
      RGB_width (int): Camera RGB stream width in pixels.
      RGB_height (int): Camera RGB stream height in pixels.
      RGB_fps (int): Camera RGB stream frame rate.
      Depth_width (int): Camera depth stream width in pixels.
      Depth_height (int): Camera depth stream height in pixels.
      Depth_fps (int): Camera depth stream frame rate.
      arm_controller: Instance of the robot arm controller created by ``create_arm``.
      camera_controller: Instance of the camera controller created by ``create_camera``.
      camera_calibration: Calibration vector/offsets loaded from the config (e.g., delta vector).
    """
    def __init__(
        self,
        pipeline_config: PipelineConfig
    ):
        """Initialize the pipeline from a configuration.

        Reads arm and camera parameters from ``pipeline_config``, prints a
        console divider, assigns configuration to instance attributes, and
        constructs device controllers via ``create_arm`` and ``create_camera``.

        Args:
          pipeline_config (PipelineConfig): Aggregated configuration for both
            the robot arm and the camera, including optional calibration data.
        """
        self.line = '*' * 150
        print(colored(self.line, color="yellow"))
        # Arm Config
        self.set_default_pos = pipeline_config.set_default_pos
        self.default_work_frame = pipeline_config.default_work_frame
        self.default_tool_frame = pipeline_config.default_tool_frame
        self.default_arm_joint = pipeline_config.default_arm_joint
        self.have_gripper = pipeline_config.have_gripper
        self.ip = pipeline_config.ip
        self.port = pipeline_config.port
        self.level = pipeline_config.level
        self.mode = pipeline_config.mode

        # Camera Config
        self.RGB_width = pipeline_config.RGB_width
        self.RGB_height = pipeline_config.RGB_height
        self.RGB_fps = pipeline_config.RGB_fps
        self.Depth_width = pipeline_config.Depth_width
        self.Depth_height = pipeline_config.Depth_height
        self.Depth_fps = pipeline_config.Depth_fps
        
        # Arm and Camera Instance
        self.arm_controller = self.create_arm()
        self.camera_controller = self.create_camera()
        self.camera_calibration = pipeline_config.delta_vector

        # UltraSearch Config
        self.search_degree = pipeline_config.search_degree

    

    def create_arm(self):
        """Create and configure the robot arm controller.

        Prints a status line, constructs the arm controller using the pipeline's
        current configuration (work/tool frames, joints, networking, etc.), and
        returns the initialized controller instance.

        Returns:
          RobotArmController: A configured arm controller instance ready for use.
        """
        print(colored("Arm Initialization: ", "light_red"))
        arm_controller = RobotArmController(
            set_default_pos=self.set_default_pos,
            default_work_frame=self.default_work_frame,
            default_tool_frame=self.default_tool_frame,
            default_arm_joint=self.default_arm_joint,
            have_gripper = self.have_gripper,
            ip=self.ip,
            level=self.level,
            mode=self.mode
            )
        print(colored(self.line, color="yellow"))
        return arm_controller

    def create_camera(self):
        """Create and configure the camera controller.

        Prints a status line, constructs the camera controller using the RGB and
        depth stream parameters from the pipeline configuration, and returns the
        initialized controller instance.

        Returns:
          Gemini335Controller: A configured camera controller instance.
        """
        print(colored("Camera Initialization: ", "light_red"))
        camera_controller = Gemini335Controller(
            RGB_width=self.RGB_width,
            RGB_height=self.RGB_height,
            RGB_fps=self.RGB_fps,
            Depth_width=self.Depth_width,
            Depth_height=self.Depth_height,
            Depth_fps=self.Depth_fps,
        )
        print(colored(self.line, color="yellow"))
        return camera_controller
        
    def disconnect(self):
        """Disconnect both the arm and camera controllers.

        Calls ``disconnect()`` on the arm and camera controllers to release
        resources and close active connections.
        """
        self.arm_controller.disconnect()
        self.camera_controller.disconnect()

    # ================================ Algorithms ================================
    def camera_point_to_tool(self, point_camera: Sequence[float]) -> np.ndarray:
        """Convert a 3D point from the camera frame to the tool frame.

        Applies a fixed translation after converting inputs to NumPy arrays:
        ``point_tool = point_camera - camera_calibration``.

        Args:
        point_camera (Sequence[float]): 3D point ``[x, y, z]`` in meters (camera frame).

        Returns:
        np.ndarray: Shape ``(3,)`` 3D point ``[x, y, z]`` in meters (tool frame).
        """
        pc = np.asarray(point_camera, dtype=float).reshape(3)
        calib = np.asarray(self.camera_calibration, dtype=float).reshape(3)
        point_tool = pc - calib
        return point_tool
            

    def camera_point_to_base(self, point_camera: Sequence[float]) -> np.ndarray:
        """Convert a 3D point from the camera frame to the robot base frame.

        Performs camera→tool via :meth:`camera_point_to_tool`, then tool→base via
        ``arm_controller.tool_point_to_base``. All intermediate values are converted
        to NumPy arrays and the result is returned as an array.

        Args:
        point_camera (Sequence[float]): 3D point ``[x, y, z]`` in meters (camera frame).

        Returns:
        np.ndarray: Shape ``(3,)`` 3D point ``[x, y, z]`` in meters (base frame).
        """
        point_tool = self.camera_point_to_tool(point_camera)  # (3,)
        point_base = np.asarray(self.arm_controller.tool_point_to_base(point_tool), dtype=float).reshape(3)
        return point_base
    
    # Ultrasound search
    def ultrasound_search(self, search_degree = -1, v: int = 5, r: int = 0, connect: int = 0, block: int = 1):
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
        if search_degree != -1:
            self.search_degree = search_degree
        delta = np.deg2rad(self.search_degree)

        # --- read current pose (for logging only) ---
        current_pose = self.arm_controller.get_current_end_pose()
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
            [0.0, 0.0, 0.0,  0.0,  0.0,  +delta],
            [0.0, 0.0, 0.0,  0.0,  0.0,  -delta],
        ], dtype=float)

        # --- execute pattern ---
        for i, off in enumerate(offsets, start=1):
            offset = current_pose + off
            status = self.arm_controller.movej_p(
                pose=offset,
                v=v,
                r=r,
                connect=connect,
                block=block
            )
        print()