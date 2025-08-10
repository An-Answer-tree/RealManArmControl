from dataclasses import dataclass, asdict
from realmanarmcontrol.base.rm_ctypes_wrap import *

@dataclass(frozen=True)
class ArmDefault:
    # TCP Connection 
    ip: str = "192.168.1.18"
    port: int = 8080
    
    # level (int): log out level，default '3'
    # - 0: debug mode
    # - 1: info mode
    # - 2: warning mode
    # - 3: error mode
    level: int = 3

    # mode (int)
    # - 0: single thread mode
    # - 1: duel threads mode
    # - 2: third threads mode
    mode: int = 2

    # Defautl Tool Frame
    default_tool_frame = {
        "name": "Ultrasound",
        "pose": [0, 0, 0.173, 0, 0, 0],
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

    # Default Arm Jonit
    default_arm_joint = [0.0, 45.0, -100.0, 0.0, -90.0, 0.0]

