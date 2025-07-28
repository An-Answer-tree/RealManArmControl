import cv2
import numpy as np
import os
import time
from termcolor import colored
from typing import Tuple, Optional

# from pyorbbecsdk import *
from pyorbbecsdk import Config, Pipeline, VideoStreamProfile, AlignFilter, PointCloudFilter
from pyorbbecsdk import FrameSet, ColorFrame, DepthFrame
from pyorbbecsdk import OBSensorType, OBFormat, OBStreamType
from pyorbbecsdk import OBError

from realmanarmcontrol.sensor.utils import frame_to_bgr_image
from realmanarmcontrol.sensor.config import Gemini335Config


class Gemini335Controller:
    """
    Controller for ORbbec Gemini 355 RGBD Camera.
    Containing: 
        1. Camera parameters initialization
        2. Operation Function
        3. Utils Function
    """
    def __init__(
        self,
        RGB_width: int = Gemini335Config.RGB_width,
        RGB_height: int = Gemini335Config.RGB_height,
        RGB_fps: int = Gemini335Config.RGB_fps,
        Depth_width: int = Gemini335Config.Depth_width,
        Depth_height: int = Gemini335Config.Depth_height,
        Depth_fps: int = Gemini335Config.Depth_fps,
    ):
        
        # Create Camera Config and Pipeline
        self.config = Config()
        self.pipeline = Pipeline()
        # Get Device SN ID
        self.device = self.pipeline.get_device()
        self.device_info = self.device.get_device_info()
        self.serial_number = self.device_info.get_serial_number()
        # Get filter for post-process
        self.depth_sensor = self.pipeline.get_device().get_sensor(OBSensorType.DEPTH_SENSOR)
        self.filter_list = self.depth_sensor.get_recommended_filters()

        # Enable Depth & Colored image sync
        self.pipeline.enable_frame_sync()


        # Init RGB Camera Setting
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

            try:
                color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(RGB_width, 
                                                                                          RGB_height, 
                                                                                          OBFormat.RGB, 
                                                                                          RGB_fps)
            except OBError as e:
                print(colored(e, "red"))
                color_profile = profile_list.get_default_video_stream_profile()

            self.config.enable_stream(color_profile)
        except Exception as e:
            print(colored(e, "red"))
            return
        
        # Init Depth Camera Setting
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                depth_profile: VideoStreamProfile = profile_list.get_video_stream_profile(Depth_width, 
                                                                                          Depth_height, 
                                                                                          OBFormat.Y16, 
                                                                                          Depth_fps)
            except OBError as e:
                print(colored(e, "red"))
                depth_profile = profile_list.get_default_video_stream_profile()

            self.config.enable_stream(depth_profile)
        except Exception as e:
            print(colored(e, "red"))
            return
        
        # Show Camera Config
        print(colored("\n✓ Camera Initialization Complete", "green"))

        RGB_config = f"RGB Camera Config: {color_profile.get_width()}x{color_profile.get_height()} @ {color_profile.get_fps()} FPS | Format: {color_profile.get_format()}"
        Depth_config = f"Depth Camera Config: {depth_profile.get_width()}x{depth_profile.get_height()} @ {depth_profile.get_fps()} FPS | Format: {depth_profile.get_format()}"
        print(colored(f"{RGB_config}\n{Depth_config}", color="cyan"))

        # Start Camera
        self.pipeline.start(self.config)
        # Create Depth & Colored images align filter
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)


    def _save_depth_frame(self, frame: DepthFrame):
        """
        Save depth_frame shotted by camera to uint16 png

        Args:
            frame (DepthFrame): Raw data recorded by depth camera

        Returns:
            data [np.uint16]: depth graph (mm)
        """
        if frame is None:
            return
        width = frame.get_width()
        height = frame.get_height()
        timestamp = frame.get_timestamp()
        scale = frame.get_depth_scale()
        depth_format = frame.get_format()

        if depth_format != OBFormat.Y16:
            print(colored("depth format is not Y16", "red"))
            return
        
        data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        data = data.reshape((height, width))

        save_image_dir = os.path.join(Gemini335Config.saved_path, "depth_images")
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir, exist_ok=True)
        filename = save_image_dir + "/depth_{}x{}_{}.png".format(width, height, timestamp)

        cv2.imwrite(filename, data)
        print(colored(f"\nDepth image saved({data.shape}):\n{filename}", "yellow"))

        return data


    def _save_color_frame(self, frame: ColorFrame):
        if frame is None:
            return
        
        width = frame.get_width()
        height = frame.get_height()
        timestamp = frame.get_timestamp()

        save_image_dir = os.path.join(Gemini335Config.saved_path, "RBG_images")
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir, exist_ok=True)
        filename = save_image_dir + "/RBG_{}x{}_{}.png".format(width, height, timestamp)

        image = frame_to_bgr_image(frame)
        if image is None:
            print(colored("\nfailed to convert frame to image", "red"))
            return
        
        cv2.imwrite(filename, image)
        print(colored(f"\nRGB image saved({image.shape}):\n{filename}", "yellow"))

        return image
    
    def _save_pointcloud(self, frames):
        # Create point cloud filter
        point_cloud_filter = PointCloudFilter()
        # Apply the alignment filter
        frame = self.align_filter.process(frames)

        # Apply the point cloud filter
        point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)
        point_cloud_frame = point_cloud_filter.process(frame)
        points = point_cloud_filter.calculate(point_cloud_frame)



    def reboot(self, device):
        device.reboot()
        print(colored(f"\nReboot the device: {self.serial_number}", "light_red"))
    
    def disconnect(self):
        """
        Disconnect from the camera.

        Returns:
            None
        """
        self.pipeline.stop()
        print(colored("\n!!!Successfully disconnected from the camera!!!", "red"))
    
    def get_profile_configs(self, sensor_type: OBSensorType):
        """
        List which config can be used under current Sensor Type.
        Print "Width x Height @ FPS | Format" that can be used.

        Args:
            pipeline (Pipeline): Camera Pipline
            sensor_type (OBSensorType): Which sensor type to search
                - OBSensorType.COLOR_SENSOR,
                - OBSensorType.DEPTH_SENSOR,
                - OBSensorType.IR_SENSOR,
                - OBSensorType.LEFT_IR_SENSOR,
                - OBSensorType.RIGHT_IR_SENSOR,
                - OBSensorType.ACCEL_SENSOR, 
                - OBSensorType.GYRO_SENSOR, 
        """
        profile_list = self.pipeline.get_stream_profile_list(sensor_type)

        for i in range(profile_list.get_count()):
            profile = profile_list.get_stream_profile_by_index(i)

            if isinstance(profile, VideoStreamProfile):
                width = profile.get_width()
                height = profile.get_height()
                fps = profile.get_fps()
                fmt = profile.get_format()

                line = f" - {width}x{height} @ {fps} FPS | Format: {fmt}"
                print(colored(line, "cyan"))
        print()

    def get_depth(
        self,
        *,
        depth_img_path: str,
        depth_data: Optional[np.ndarray] = None,
        point: Tuple[int, int]
    ) -> Tuple[int, int, int]:
        """
        Get the depth (in millimeters) at a specific pixel coordinate.

        Args:
            depth_img_path (str): Path to a saved 16‑bit PNG depth image.
            depth_data (Optional[np.ndarray]): A pre‑loaded depth array (dtype=uint16).
            point (Tuple[int, int]): Pixel coordinates (u, v).

        Returns:
            Tuple[int, int, int]: (u, v, depth_mm) where depth_mm is the depth value in millimeters.
        """
        u, v = point

        # If no depth array is provided, load it from file
        if depth_data is None:
            # Read image unchanged to preserve 16‑bit values
            depth_data = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
            if depth_data is None:
                raise IOError(f"Failed to load depth image from '{depth_img_path}'")

        # Verify the image is single‑channel uint16
        if depth_data.dtype != np.uint16 or depth_data.ndim not in (2,):
            raise ValueError("Depth data must be a 2D uint16 array")

        height, width = depth_data.shape

        # Ensure the point is within image bounds
        if not (0 <= u < width and 0 <= v < height):
            raise ValueError(f"Point ({u}, {v}) is outside image bounds ({width}×{height})")

        # Retrieve the depth value at (v, u)
        depth_mm = int(depth_data[v, u])

        return (u, v, depth_mm)
    
    def depth_post_process(self, depth_frame):
        for post_filter in self.filter_list:
            if post_filter.is_enabled():
                new_depth_frame = post_filter.process(depth_frame)
                depth_frame = new_depth_frame.as_depth_frame()
        return depth_frame

    def take_photo(self):
        """
        Capture one RGB+Depth frame from the Gemini 355 camera.

        Args:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - RGB_image: BGR color image (png) as a NumPy array.
                - Depth_graph: Depth Graph (png) as a NumPy array.
        """
        frames = None

        while True:
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                continue

            frames = self.align_filter.process(frames)
            if not frames:
                continue
            
            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame == None or depth_frame == None:
                continue
            else:
                break
        
        # Save RGB Image
        RGB_image = self._save_color_frame(color_frame)
        # Post-Process and Save Depth Image
        depth_frame = self.depth_post_process(depth_frame)
        Depth_graph = self._save_depth_frame(depth_frame)

        return (RGB_image, Depth_graph)
