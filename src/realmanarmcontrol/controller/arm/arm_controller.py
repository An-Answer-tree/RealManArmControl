import math
from math import sin, cos
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import time
from termcolor import colored
from typing import Sequence, Tuple

from .arm_config import ArmDefault
from realmanarmcontrol.base.rm_robot_interface import *
from realmanarmcontrol.base.rm_ctypes_wrap import *


class RobotArmController:
    # =====================Connect Arm and Initialize=====================
    def __init__(
            self, 
            set_default_pos: bool = ArmDefault.set_default_pos,
            default_work_frame: dict = ArmDefault.default_work_frame,
            default_tool_frame: dict = ArmDefault.default_tool_frame,
            default_arm_joint: list = ArmDefault.default_arm_joint,
            have_gripper: bool = ArmDefault.have_gripper,
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
            have_gripper: Whether have a gripper
            ip (str): IP address of the robot arm.
            port (int): Port number.
            level (int, optional): Connection level. Defaults to 3.
            mode (int, optional): Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
        """

        # Create Connection
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)
        self.have_gripper = have_gripper

        # Gripper Config
        self.set_gripper_min_max(0, 150)
        
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


    # ==========================================Tool_Frames_Function==========================================
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


    # ==========================================Arm_Status_Function==========================================
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
        # print("Current end-effector pose:", pose)
        return pose
    

    # ==========================================Gripper Status==========================================
    def get_gripper_status(self) -> dict:
        """Retrieve the current gripper status from the controller.

        Verifies that a gripper is present and queries the controller via
        ``rm_get_gripper_state()``. On success, returns the controller-reported
        state dictionary. On failure, raises a descriptive error instead of
        relying on ``assert`` (which can be optimized out with ``-O``).

        Returns:
            dict: The gripper state as returned by ``rm_get_gripper_state()``.

        Raises:
            RuntimeError: If no gripper is installed or the controller returns a
                non-zero status code. The error message includes a human-readable
                explanation mapped from the status code.
        """
        if not getattr(self, "have_gripper", False):
            raise RuntimeError(colored("Arm doesn't have a gripper", "red"))

        # Controller returns (tag, state_dict), where tag==0 means success.
        tag, status = self.robot.rm_get_gripper_state()

        if tag != 0:
            # Map SDK status codes to readable messages.
            code_map = {
                1: "Controller reported failure (invalid parameter or arm error).",
                -1: "Send failed (communication issue).",
                -2: "Receive failed or controller timeout.",
                -3: "Response parse failed (malformed or incomplete data).",
                -4: "Operation timed out.",  # present in other APIs; keep for completeness
            }
            msg = code_map.get(tag, "Unknown error.")
            self.have_gripper = False
            print(f"Failed to get gripper state (code {tag}): {msg}")

        return status


    # ==========================================Algo_Function==========================================
    def get_euler_towards_object(self,
                                eef_pos: np.ndarray,
                                target_pos: np.ndarray,
                                up: list = [0, 0, 1]):
        """Compute intrinsic-XYZ Euler angles for a look-at tool orientation.

        Constructs a right-handed orthonormal frame where:
        - The local +Z axis points from ``eef_pos`` to ``target_pos``.
        - The local +Y axis is the **negative** of the projection of ``up`` onto the
            plane orthogonal to +Z (i.e., as close as possible to ``-up`` while
            remaining perpendicular to +Z).
        - The local +X axis is computed as ``X = Y × Z`` to guarantee a proper
            rotation (determinant +1).

        The function returns the intrinsic XYZ Euler angles (radians) that realize
        this orientation, along with the 3×3 rotation matrix whose columns are
        ``[X, Y, Z]`` expressed in the base frame.

        Args:
            eef_pos: Current end-effector 3D position ``[x, y, z]`` (meters).
            target_pos: 3D position to look at ``[x, y, z]`` (meters).
            up: World-up reference vector (default ``[0, 0, 1]``). It can be non-unit.

        Returns:
            tuple[list[float], np.ndarray]: A pair ``(euler_xyz, R_base_tool)`` where
            ``euler_xyz`` is ``[rx, ry, rz]`` (radians, intrinsic XYZ) and
            ``R_base_tool`` is a ``(3, 3)`` rotation matrix with columns ``[X, Y, Z]``.

        Raises:
            ValueError: If positions are invalid or nearly identical (undefined look
                direction).
            RuntimeError: If the projected up-vector is (near) collinear with Z, or if
                the resulting frame is degenerate.

        Notes:
            - Uses Gram–Schmidt to orthogonalize the up-vector against Z.
            - Ensures a proper right-handed rotation by defining ``X = Y × Z``.
        """
        # -- sanitize inputs --
        eef_pos = np.asarray(eef_pos, dtype=float).reshape(3)
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        up_vec = np.asarray(up, dtype=float).reshape(3)

        if not (np.all(np.isfinite(eef_pos)) and np.all(np.isfinite(target_pos)) and np.all(np.isfinite(up_vec))):
            raise ValueError(colored("Non-finite values detected in inputs (NaN/Inf).", "red"))

        if np.allclose(eef_pos, target_pos, atol=1e-6):
            raise ValueError(colored("eef_pos and target_pos coincide; look direction is undefined.", "red"))

        # -- Z axis: point to target --
        z_axis = target_pos - eef_pos
        zn = np.linalg.norm(z_axis)
        if zn < 1e-9:
            raise RuntimeError(colored("Direction vector is near zero; cannot define +Z.", "red"))
        z_axis = z_axis / zn

        # -- Y axis: negative of up's projection onto plane orthogonal to Z --
        # proj = (up · z) z ;  y_raw = -(up - proj)
        proj = np.dot(up_vec, z_axis) * z_axis
        x_axis = (up_vec - proj)
        xn = np.linalg.norm(x_axis)
        if xn < 1e-9:
            # keep your original error-handling logic (no fallback): raise
            raise RuntimeError(colored("The Z-axis is collinear with the vertical direction (degenerate up).", "red"))
        x_axis = x_axis / xn

        # -- X axis: ensure right-handed proper rotation: X = Y × Z --
        y_axis = np.cross(z_axis, x_axis)
        yn = np.linalg.norm(y_axis)
        if yn < 1e-9:
            raise RuntimeError(colored("Failed to construct a valid X axis (degenerate frame).", "red"))
        y_axis = y_axis / yn

        # -- rotation matrix with columns [X, Y, Z] --
        R_base_tool = np.stack([x_axis, y_axis, z_axis], axis=1)

        # -- Euler (intrinsic XYZ, radians) --
        euler_xyz = R.from_matrix(R_base_tool).as_euler('xyz', degrees=False).tolist()

        # colored success output
        print(colored("✓ look-at orientation computed.", "green"),
            colored(f"Euler(XYZ, rad)={np.array(euler_xyz)}\n", "cyan"))

        return euler_xyz, R_base_tool

    def tool_point_to_base(self, point_tool: Sequence[float]) -> np.ndarray:
        """Transform a point from the tool frame to the base frame.

        Uses the current end-effector pose returned by ``get_current_end_pose()``,
        which is assumed to be ``[x, y, z, rx, ry, rz]`` where the Euler angles
        are intrinsic XYZ (rotate about X, then Y, then Z) in radians. The
        rotation matrix is constructed with pure NumPy and applied as
        ``p_base = R_base_tool @ p_tool + t_base``.

        Args:
            point_tool: Length-3 sequence ``[xt, yt, zt]`` in meters expressed in
                the tool frame.

        Returns:
            np.ndarray: Shape ``(3,)`` array of the point in meters expressed in
            the base frame.

        Raises:
            RuntimeError: If ``get_current_end_pose()`` does not return at least
                six values.
            ValueError: If ``point_tool`` is not length 3 or contains NaN/Inf.
        """
        # Validate input
        if not hasattr(point_tool, "__len__") or len(point_tool) != 3:
            raise ValueError("point_tool must be a length-3 sequence: [xt, yt, zt].")
        p_tool = np.asarray(point_tool, dtype=float).reshape(3)
        if not np.all(np.isfinite(p_tool)):
            raise ValueError(f"Invalid point_tool (nan/inf): {p_tool}")

        # Read current end-effector pose: [x, y, z, rx, ry, rz]
        pose = self.get_current_end_pose()
        if pose is None or len(pose) < 6:
            raise RuntimeError("get_current_end_pose() must return [x, y, z, rx, ry, rz].")

        x0, y0, z0, rx, ry, rz = map(float, pose[:6])
        t_base = np.array([x0, y0, z0], dtype=float)

        # Build rotation from intrinsic XYZ Euler angles (radians) using pure NumPy.
        # Intrinsic XYZ is equivalent to extrinsic ZYX, so R = Rz(rz) @ Ry(ry) @ Rx(rx).
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1.0, 0.0, 0.0],
                    [0.0,  cx, -sx],
                    [0.0,  sx,  cx]], dtype=float)
        Ry = np.array([[ cy, 0.0,  sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0,  cy]], dtype=float)
        Rz = np.array([[ cz, -sz, 0.0],
                    [ sz,  cz, 0.0],
                    [0.0, 0.0, 1.0]], dtype=float)

        rot_base_tool = Rz @ Ry @ Rx  # (3,3)

        # Rotate then translate: p_base = R * p_tool + t
        p_base = rot_base_tool @ p_tool + t_base
        return p_base


    # ==========================================Arm_Moving_Function==========================================
    def movej(self, joint: list[float], v: int = 30, r: int = 0, connect: int = 0, block: int = 1) -> int:
        """Perform joint space motion with default parameters and raise error on failure.

        This is a wrapper for ``rm_movej`` that moves the robot arm to the specified
        joint angles in joint space. It uses default parameters for speed, blend
        radius, connection flag, and blocking behavior. Non-zero status codes will
        raise a RuntimeError with a descriptive message.

        Args:
            joint (list[float]): Target joint angles in degrees for each joint.
            v (int, optional): Speed scaling factor (1–100). Defaults to 30.
            r (int, optional): Blend radius scaling factor (0–100). Defaults to 0.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Blocking mode. Defaults to 1.

        Returns:
            int: Status code (0 for success).

        Raises:
            RuntimeError: If the status code is non-zero, with a message describing the failure.
        """
        code = self.robot.rm_movej(joint, v, r, connect, block)

        if code != 0:
            error_map = {
                1: "Controller error: invalid parameter or arm state error.",
                -1: "Send error: data send failure.",
                -2: "Receive error: data receive failure or controller timeout.",
                -3: "Parse error: response format invalid or incomplete.",
                -4: "Device mismatch: current device is not in joint mode.",
                -5: "Timeout: single-thread mode timed out.",
            }
            message = error_map.get(code, f"Unknown error with status code {code}.")
            raise RuntimeError(f"movej failed: {message}")

        return code
    
    def movej_p(self,
                pose: list[float],
                v: int = 30,
                r: int = 0,
                connect: int = 0,
                block: int = 1) -> int:
        """Move in joint-space to a Cartesian pose with light validation and colored logs.

        Args:
            pose: Target pose ``[x, y, z, rx, ry, rz]`` (meters, radians; intrinsic XYZ).
            v: Speed percentage ``1..100``. Defaults to 30.
            r: Blend percentage ``0..100`` (0=disabled). Defaults to 0.
            connect: Trajectory connection flag ``0|1``. Defaults to 0.
            block: Blocking/timeout setting (see controller docs). Defaults to 1.

        Returns:
            int: Status code from the controller (0 on success).
        """
        # --- minimal normalization ---
        arr = np.asarray(pose, dtype=float).flatten()
        if arr.size != 6 or not np.all(np.isfinite(arr)):
            raise ValueError("pose must be 6 finite numbers [x,y,z,rx,ry,rz].")

        v, r, connect, block = int(v), int(r), int(connect), int(block)

        # --- delegate ---
        try:
            status = self.robot.rm_movej_p(arr.tolist(), v, r, connect, block)
        except AttributeError as e:
            print(colored("✗ movej_p failed: rm_movej_p missing.", "red"))
            raise

        # --- compact reporting ---
        if status == 0:
            print(colored("✓ movej_p executed.", "green"),
                colored(f"pose={arr.tolist()}, v={v}, r={r}, connect={connect}, block={block}", "cyan"))
        else:
            reason = {
                1:  "invalid parameters or robot state",
                -1: "send failure",
                -2: "receive timeout",
                -3: "response parse error",
                -4: "device verification failed",
                -5: "single-thread timeout",
            }.get(status, "unknown error")
            print(colored(f"✗ movej_p failed (code={status})", "red"),
                colored(f"reason: {reason}", "yellow"))
        return status

    def move_follow(self, pose: list[float]) -> None:
        """Stream a Cartesian pose in follow mode (xyz + quaternion or Euler).

        Thin wrapper that forwards the target pose to ``rm_movep_follow`` for
        follow-style Cartesian motion. The pose can be either 6D (Euler) or
        7D (quaternion).

        Args:
        pose (list[float]): Target pose.
            - If ``len(pose) == 7``: ``[x, y, z, qx, qy, qz, qw]`` (meters, unit quaternion).
            - If ``len(pose) == 6``: ``[x, y, z, rx, ry, rz]`` (meters, radians).

        Returns:
        None
        """
        ret = self.robot.rm_movep_follow(pose)

        if ret == 0:
            print("✓ MoveP (follow) executed successfully.")
            print("Target pose:", pose)
        else:
            error_msg = {
                1:  "Controller returned false: invalid parameters or robot state error.",
                -1: "Data send failure: communication issue (network/serial).",
            }
            tips = {
                1:  ("Validate pose length (6 or 7) and units (meters/radians); "
                    "ensure quaternion order & normalization; check v/limits in controller; "
                    "verify the robot is not in E-Stop/alarm state."),
                -1: ("Check physical link and drivers; ensure stable update cycle; "
                    "reduce streaming rate; inspect bandwidth/packet loss."),
            }
            msg = error_msg.get(ret, "Unknown error code (not documented by SDK).")
            hint = tips.get(ret, "Check parameter ranges, comms link, and controller state.")

            print(f"✗ MoveP (follow) failed. Code: {ret}")
            print(f"Reason: {msg}")
            print(f"Suggestion: {hint}")

    def movej_p_look_at(self, target_pos: list, look_at_pos: list, up=[0, 0, 1], v=5, r=0, connect=0, block=1) -> None:
        """Move to a target position while keeping the end-effector Z-axis pointing to a point.

        Computes an orientation that makes the end-effector's local Z-axis point to
        ``look_at_pos`` from ``target_pos``, concatenates it with the position to form
        a 6D pose, and then executes a joint-space motion via ``rm_movej_p``.
        Control flow remains identical to your original implementation; only output
        messages were improved.

        Args:
        target_pos (list[float]): Target position [x, y, z] in meters.
        look_at_pos (list[float]): World position the Z-axis should point toward.
        v (int, optional): Speed percentage [1, 100]. Defaults to 30.
        r (int, optional): Blend radius percentage [0, 100]; 0 disables blending. Defaults to 0.
        connect (int, optional): Trajectory connection flag.
            - 0: Execute immediately.
            - 1: Co-plan with the next trajectory and do not execute immediately.
            Defaults to 0.
        block (int, optional): Blocking behavior.
            - Multi-threaded: 0 non-blocking; 1 blocking until arrival/failure.
            - Single-threaded: 0 non-blocking; >0 blocking with timeout (seconds).
            Defaults to 1.

        Returns:
        None

        Side Effects:
        Sends a motion command and prints human-readable result messages.

        """
        # Compute Euler orientation (radians) that makes the local Z-axis "look at" the target point.
        rx, ry, rz = self.get_euler_towards_object(eef_pos=target_pos, target_pos=look_at_pos, up=up)[0]
        pose = list(target_pos) + [float(rx), float(ry), float(rz)]

        # Execute joint-space motion
        result = self.robot.rm_movej_p(pose, v=v, r=r, connect=connect, block=block)

        if result == 0:
            print(colored("✓ MoveJ command executed successfully.", color="green"))
            print(colored("Target pose:", "cyan"), colored(pose, "cyan"))
        else:
            # Enhanced error mapping and troubleshooting tips (control flow unchanged)
            error_msg = {
                1:  "Controller returned false: invalid parameters or robot state error.",
                -1: "Data send failure: communication issue (network/serial/bandwidth).",
                -2: "Data receive failure or timeout: controller did not respond in time.",
                -3: "Response parsing failure: malformed or incomplete data.",
                -4: "Device verification failed: current 'in-position' device is not joint.",
                -5: "Single-threaded timeout: no response within the 'block' timeout."
            }
            tips = {
                1:  "Validate pose length (6) and units (meters/radians); check v∈[1,100], r∈[0,100], connect∈{0,1}, block mode; "
                    "ensure the robot is not in E-Stop/alarm.",
                -1: "Check physical link and network quality; if remote, reduce send rate or inspect bandwidth/packet loss.",
                -2: "Increase 'block' timeout in single-threaded mode; check controller load/queue; ensure it isn't busy.",
                -3: "Verify controller version/protocol; capture traffic to see if the response is truncated or mixed.",
                -4: "Confirm the controller is configured for joint 'in-position' device mode.",
                -5: "Increase the 'block' timeout; or in multi-threaded mode use blocking=1; investigate abnormal execution time."
            }
            msg = error_msg.get(result, "Unknown error code (not documented).")
            hint = tips.get(result, "Check parameter ranges, comms link, and controller state.")

            print(colored(f"✗ MoveJ command failed. Code: {result}", "red"))
            print(colored(f"Reason: {msg}", "yellow"))
            print(colored(f"Suggestion: {hint}", "magenta"))

        
    # ==========================================Gripper Control==========================================
    def set_gripper_min_max(self, min, max) -> int:
        """Set gripper travel limits (minimum and maximum opening) with colored output.

        This method validates inputs and delegates to ``rm_set_gripper_route``.
        The controller persists the limits across power cycles.

        Args:
            min (int): Minimum opening (inclusive), valid range ``0..1000`` (dimensionless).
            max (int): Maximum opening (inclusive), valid range ``0..1000`` (dimensionless).

        Returns:
            int: Status code returned by the controller.
                - ``0``: Success.
                - ``1``: Controller returned false (invalid parameters or robot state error).
                - ``-1``: Data send failure.
                - ``-2``: Data receive failure or controller timeout.
                - ``-3``: Response parsing failure.
                - ``-4``: Timeout.

        Raises:
            TypeError: If inputs are not numeric.
            ValueError: If inputs are out of range or ``min > max``.
            RuntimeError: If the SDK method is unavailable on this instance.
        """
        # ---- input validation ----
        if not (isinstance(min, (int, float)) and isinstance(max, (int, float))):
            raise TypeError("min and max must be numeric (int or float).")

        min_route = int(min)
        max_route = int(max)

        if not (0 <= min_route <= 1000 and 0 <= max_route <= 1000):
            raise ValueError(
                f"min/max out of range: min={min_route}, max={max_route} (expected 0..1000)."
            )
        if min_route > max_route:
            raise ValueError(f"min must be <= max (got min={min_route}, max={max_route}).")

        # ---- delegate to SDK ----
        try:
            status = self.robot.rm_set_gripper_route(min_route, max_route)
        except AttributeError as e:
            hdr = colored("✗ set_gripper_min_max failed", "red")
            print(hdr)
            print(colored(
                "Reason: rm_set_gripper_route is not available on this instance.",
                "yellow"
            ))
            print(colored(
                "Suggestion: Ensure the SDK/driver is initialized and the method is bound "
                "(e.g., self.robot.rm_set_gripper_route or self.rm_set_gripper_route).",
                "magenta"
            ))
            raise RuntimeError(
                "rm_set_gripper_route is not available on this instance."
            ) from e

        # ---- human-readable diagnostics ----
        messages = {
            0:  "Success.",
            1:  "Controller returned false: invalid parameters or robot state error.",
            -1: "Data send failure (communication issue).",
            -2: "Data receive failure or controller timeout.",
            -3: "Response parsing failure (malformed/incomplete data).",
            -4: "Operation timed out.",
        }
        suggestions = {
            0:  f"Limits persisted: min={min_route}, max={max_route}.",
            1:  ("Verify: 0<=min<=max<=1000; controller not in E-Stop/alarm; gripper enabled "
                "and homed; no motion/hold interlocks; correct device selected."),
            -1: "Check the physical link/network quality and SDK connection handle.",
            -2: "Increase controller timeout; ensure controller is responsive and not busy.",
            -3: "Check controller firmware/protocol compatibility; update SDK if needed.",
            -4: "Retry with a larger timeout; ensure controller load is normal.",
        }

        if status == 0:
            print(
                colored("✓ set_gripper_min_max ok", "green"),
                colored(f"(min={min_route}, max={max_route})", "cyan")
            )
        else:
            print(colored(f"✗ set_gripper_min_max failed (code={status})", "red"))
            print(colored(f"  Reason: {messages.get(status, 'Unknown error code.')}", "yellow"))
            print(colored(f"  Suggestion: {suggestions.get(status, 'Check parameters, controller state, and comms.')}", "magenta"))

        return status


    def release_gripper(self, speed: int, block: bool = True, timeout: int = 5) -> None:
        """Open the gripper to its maximum position.

        Commands the gripper to fully open at a specified speed. Can operate in
        blocking or non-blocking mode, with an optional timeout. Raises an error
        if the arm has no gripper or the release command fails.

        Args:
            speed (int): Opening speed in the range [1, 1000], unitless.
            block (bool, optional): Whether to operate in blocking mode.
                If True, waits until the controller reports that the gripper
                has reached its maximum opening or until timeout. If False,
                returns immediately after sending the command. Defaults to True.
            timeout (int, optional): Timeout in seconds. In blocking mode,
                specifies how long to wait for the gripper to reach the maximum
                opening. In non-blocking mode, ``0`` means return immediately,
                other positive values wait for the "command accepted" response.
                Defaults to 5.

        Raises:
            RuntimeError: If the arm has no gripper or if the controller returns a
                non-zero status code. The exception message includes the error code
                and a human-readable explanation.

        """
        if not getattr(self, "have_gripper", False):
            raise RuntimeError(colored("Arm doesn't have a gripper", "red"))

        tag = self.robot.rm_set_gripper_release(speed, block, timeout)

        if tag != 0:
            code_map = {
                1: "Controller reported failure (invalid parameter or arm error).",
                -1: "Send failed (communication issue).",
                -2: "Receive failed or controller timeout.",
                -3: "Response parse failed (malformed or incomplete data).",
                -4: "Operation timed out.",
            }
            msg = code_map.get(tag, "Unknown error.")
            raise RuntimeError(f"Failed to release gripper (code {tag}): {msg}")
        
    def set_gripper_pos(self, position: int, block: bool = True, timeout: int = 5) -> None:
        """Move the gripper to a specific opening position.

        Commands the gripper to reach a target opening position. Can operate in
        blocking or non-blocking mode, with an optional timeout. Raises an error
        if the arm has no gripper or the command fails.

        Args:
            position (int): Target gripper opening position in the range [1, 1000],
                unitless.
            block (bool, optional): Whether to operate in blocking mode.
                If True, waits until the controller reports that the gripper has
                reached the target position or until timeout. If False, returns
                immediately after sending the command.
            timeout (int, optional): Timeout in seconds. In blocking mode,
                specifies how long to wait for the gripper to reach the target
                position. In non-blocking mode, ``0`` means return immediately;
                other positive values wait for the "command accepted" response.
                Defaults to 5.

        Raises:
            RuntimeError: If the arm has no gripper or if the controller returns a
                non-zero status code. The exception message includes the error code
                and a human-readable explanation.
        """
        if not getattr(self, "have_gripper", False):
            raise RuntimeError(colored("Arm doesn't have a gripper", "red"))

        tag = self.robot.rm_set_gripper_position(position, block, timeout)

        if tag != 0:
            code_map = {
                1: "Controller reported failure (invalid parameter or arm error).",
                -1: "Send failed (communication issue).",
                -2: "Receive failed or controller timeout.",
                -3: "Response parse failed (malformed or incomplete data).",
                -4: "Operation timed out.",
            }
            msg = code_map.get(tag, "Unknown error.")
            raise RuntimeError(f"Failed to set gripper position (code {tag}): {msg}")
        
    def gripper_keep_pick(self, speed: int = 500, force: int = 100, block: bool = True, timeout: int = 10) -> None:
        """Perform continuous force-controlled gripping.

        Commands the gripper to close at a specified speed and maintain the given
        gripping force continuously, instead of stopping once the force threshold
        is reached. This is useful for preventing slippage during prolonged
        handling. Can operate in blocking or non-blocking mode.

        Args:
            speed (int): Gripping speed in the range [1, 1000], unitless.
            force (int): Force threshold in the range [50, 1000], unitless.
            block (bool): Whether to operate in blocking mode.
                If True, waits until the controller reports the gripper has
                reached the desired force or until timeout. If False, returns
                immediately after sending the command.
            timeout (int): Timeout in seconds. In blocking mode, specifies how long
                to wait for the gripper to reach the desired force. In non-blocking
                mode, ``0`` means return immediately; other positive values wait
                for the "command accepted" response.

        Raises:
            RuntimeError: If the arm has no gripper or if the controller returns a
                non-zero status code. The exception message includes the error code
                and a human-readable explanation.
        """
        if not getattr(self, "have_gripper", False):
            raise RuntimeError(colored("Arm doesn't have a gripper", "red"))

        tag = self.robot.rm_set_gripper_pick_on(speed, force, block, timeout)

        if tag != 0:
            code_map = {
                1: "Controller reported failure (invalid parameter or arm error).",
                -1: "Send failed (communication issue).",
                -2: "Receive failed or controller timeout.",
                -3: "Response parse failed (malformed or incomplete data).",
                -4: "Operation timed out.",
            }
            msg = code_map.get(tag, "Unknown error.")
            raise RuntimeError(f"Failed to perform continuous gripping (code {tag}): {msg}")