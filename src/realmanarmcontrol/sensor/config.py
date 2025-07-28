from dataclasses import dataclass

@dataclass
class Gemini335Config:
    """
    Paramaters for Camera Initialization
    """
    # RGB
    RGB_width = 1280
    RGB_height = 720
    RGB_fps = 30

    # Depth
    Depth_width = 1280
    Depth_height = 720
    Depth_fps = 30

    # RGBD Saved Path
    saved_path = "saved_images"

