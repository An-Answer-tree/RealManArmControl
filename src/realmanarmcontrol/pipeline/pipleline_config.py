from dataclasses import dataclass
from typing import Tuple

from realmanarmcontrol.controller.sensor. camera_config import Gemini335Config
from realmanarmcontrol.controller.arm.arm_config import ArmDefault

@dataclass(frozen=True)
class PipelineConfig(Gemini335Config, ArmDefault):
    # Eye-on-hand calibration offsets (tool frame → camera/EE), in meters.
    delta_x: float = 0
    delta_y: float = 0.0988
    # 0.171
    delta_z: float = 0.165
    delta_vector: Tuple = (delta_x, delta_y, delta_z)

    # Arm Config
    # Defautl Tool Frame
    default_tool_frame = {
        "name": "Ultrasound",
        "pose": [0.02344, 0, 0.2062, 0, 0, 1.57],
        "payload": 0.5,
        "center_of_mass": [0, 0, 0.055]
    }
    # Default Work Frame
    default_work_frame = {
        "name": "BaseRev",
        "pose": [0, 0, 0, 0, 0, 3.142]
    }
    # Set to Default when start
    set_default_pos = True
    default_arm_joint = [0.0, 45.0, -100.0, 0.0, -90.0, 0.0]
    # Whether have gripper
    have_gripper = True


    # YOLO
    model_path = "/home/user/Project/RealManArmControl/src/realmanarmcontrol/algorithms/yolo/model/best.pt"


    # UltraSound Search Config
    search_degree = 10
    