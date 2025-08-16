import cv2
import numpy as np
import os
import time
from termcolor import colored
from typing import Tuple, Optional, Sequence

from pyorbbecsdk import Config, Pipeline, VideoStreamProfile, AlignFilter, PointCloudFilter
from pyorbbecsdk import FrameSet, ColorFrame, DepthFrame
from pyorbbecsdk import OBSensorType, OBFormat, OBStreamType, OBError, OBFrameAggregateOutputMode
from pyorbbecsdk import save_point_cloud_to_ply
from pyorbbecsdk import (
    transformation2dto2d,
    transformation2dto3d,
    transformation3dto3d,
    transformation3dto2d
)

from .utils import frame_to_bgr_image
from .camera_config import Gemini335Config


class Gemini335Controller:
    """Controller for the ORBBEC Gemini 335 RGB-D camera.

    This class encapsulates device discovery, stream configuration, and basic
    pipeline control for synchronized RGB and depth streaming. It also provides
    a utility to print human-readable camera configuration information.

    Attributes:
        RGB_width: Target RGB stream width in pixels.
        RGB_height: Target RGB stream height in pixels.
        RGB_fps: Target RGB stream frame rate (frames per second).
        Depth_width: Target depth stream width in pixels.
        Depth_height: Target depth stream height in pixels.
        Depth_fps: Target depth stream frame rate (frames per second).
        config: Low-level stream configuration object.
        pipeline: Stream pipeline controller.
        color_profile: Selected RGB video stream profile, or None if unset.
        depth_profile: Selected depth video stream profile, or None if unset.
        device: Physical camera device handle.
        device_info: Device information handle.
        serial_number: Device serial number string.
        depth_sensor: Depth sensor handle.
        filter_list: Recommended post-processing filters from the depth sensor.
        align_filter: Alignment filter used to align depth to the RGB stream.
    """

    def __init__(
        self,
        RGB_width: int,
        RGB_height: int,
        RGB_fps: int,
        Depth_width: int,
        Depth_height: int,
        Depth_fps: int,
        save_path: str
    ):
        """Initializes the controller and starts the streaming pipeline.

        This constructor:
          * Stores desired RGB and depth stream parameters.
          * Creates the low-level config and pipeline objects.
          * Discovers the connected device and basic device info.
          * Fetches recommended depth post-processing filters.
          * Enables frame synchronization between RGB and depth streams.
          * Initializes RGB and depth stream profiles and enables them.
          * Prints the selected camera configuration.
          * Starts the pipeline and prepares a depth-to-color align filter.

        Args:
            RGB_width: Desired RGB width in pixels.
            RGB_height: Desired RGB height in pixels.
            RGB_fps: Desired RGB frame rate.
            Depth_width: Desired depth width in pixels.
            Depth_height: Desired depth height in pixels.
            Depth_fps: Desired depth frame rate.

        Note:
            Errors during profile selection are logged and the code falls back
            to a default stream profile provided by the SDK.
        """
        # Useful paramaters
        self.RGB_width = RGB_width
        self.RGB_height = RGB_height
        self.RGB_fps = RGB_fps
        self.Depth_width = Depth_width
        self.Depth_height = Depth_height
        self.Depth_fps = Depth_fps
        self.save_path = save_path
        self.time_stamp = None
        # Create Camera Config and Pipeline
        self.config = Config()
        self.pipeline = Pipeline()
        # Config Profile
        self.color_profile = None
        self.depth_profile = None
        # Get Device SN ID
        self.device = self.pipeline.get_device()
        self.device_info = self.device.get_device_info()
        self.serial_number = self.device_info.get_serial_number()
        # Get filter for post-process
        self.depth_sensor = self.device.get_sensor(OBSensorType.DEPTH_SENSOR)
        self.filter_list = self.depth_sensor.get_recommended_filters()
        # Utils for point cloud extraction
        self.align_filter = None
        self.point_cloud_filter = None

        # Init RGB and Depth setting
        self.init_RGB_setting()
        self.init_Depth_setting()
        
        # Print Camera Info
        self.get_camera_info()

        # Start Pipeline
        self.start_pipeline()


    # =================================Config Camera=================================
    def init_RGB_setting(self):
        """Selects and enables the RGB video stream profile.

        Attempts to acquire a video stream profile that matches the requested
        resolution, pixel format, and frame rate. If a matching profile is not
        available, logs the SDK error and falls back to the device default
        profile. The chosen profile is then enabled in the pipeline config.

        Side effects:
            Sets `self.color_profile` and enables it in `self.config`.

        Raises:
            This method does not raise. SDK errors are caught and printed.
        """
        # Init RGB Camera Setting
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)

            try:
                self.color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(self.RGB_width, 
                                                                                                self.RGB_height, 
                                                                                                OBFormat.RGB, 
                                                                                                self.RGB_fps)
            except OBError as e:
                print(colored(e, "red"))
                self.color_profile = profile_list.get_default_video_stream_profile()

            self.config.enable_stream(self.color_profile)
        except Exception as e:
            print(colored(e, "red"))
            return 

    def init_Depth_setting(self):
        """Selects and enables the depth video stream profile.

        Attempts to acquire a depth stream profile with the requested
        resolution and frame rate in Y16 format. If a matching profile is not
        available, logs the SDK error and falls back to the device default
        profile. The chosen profile is then enabled in the pipeline config.

        Side effects:
            Sets `self.depth_profile` and enables it in `self.config`.

        Raises:
            This method does not raise. SDK errors are caught and printed.
        """
        # Init Depth Camera Setting
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                self.depth_profile: VideoStreamProfile = profile_list.get_video_stream_profile(self.Depth_width, 
                                                                                                self.Depth_height, 
                                                                                                OBFormat.Y16, 
                                                                                                self.Depth_fps)
            except OBError as e:
                print(colored(e, "red"))
                self.depth_profile = profile_list.get_default_video_stream_profile()

            self.config.enable_stream(self.depth_profile)
        except Exception as e:
            print(colored(e, "red"))
            return

    def start_pipeline(self):
        """Starts the streaming pipeline and prepares depth-to-RGB alignment.

        Side effects:
            Starts `self.pipeline` with the current `self.config` and creates
            `self.align_filter` to align depth frames to the RGB stream.
        """
        # Ensure all types of frames are included in the output frameset
        self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        # Start Camera
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        # For point cloud extraction
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.point_cloud_filter = PointCloudFilter()

    def reboot(self, device):
        """
        Reboot the camera device

        Args:
            device : Device you want to reboot
        """
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


    # =================================Camera Info=================================
    def get_camera_info(self):
        """Prints and returns human-readable RGB and depth stream settings.

        Builds formatted strings describing the selected RGB and depth stream
        profiles (resolution, FPS, and pixel format), prints them to the
        console, and returns both strings.

        Returns:
            Tuple[str, str]: (`RGB_config`, `Depth_config`) describing the RGB
            and depth stream configurations.
        """
        # Show Camera Config
        print(colored("\n✓ Camera Initialization Complete", "green"))
        cp = self.color_profile
        dp = self.depth_profile
        RGB_config = (
            "RGB Camera Config: "
            f"{cp.get_width()}x{cp.get_height()} @ "
            f"{cp.get_fps()} FPS | "
            f"Format: {cp.get_format()}"
        )
        Depth_config = (
            "Depth Camera Config: "
            f"{dp.get_width()}x{dp.get_height()} @ "
            f"{dp.get_fps()} FPS | "
            f"Format: {dp.get_format()}"
        )
        print(colored(f"{RGB_config}\n{Depth_config}", color="cyan"))
        return RGB_config, Depth_config
    
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



    # =================================Private Utils Function=================================
    def _save_depth_frame(self, frame: DepthFrame):
        """
        Save depth_frame shotted by camera to uint16 png

        Args:
            frame (DepthFrame): Raw data recorded by depth camera

        Returns:
            data [np.uint16]: depth graph (mm)
        """
        # Process depth frame to depth data
        width = frame.get_width()
        height = frame.get_height()
        # scale = frame.get_depth_scale()
        data = np.frombuffer(frame.get_data(), dtype=np.uint16)
        data = data.reshape((height, width))

        # Create depth graph save directory
        save_depth_dir = os.path.join(self.save_path, "depth_images")
        os.makedirs(save_depth_dir, exist_ok=True)
        # Set file name
        filename = os.path.join(save_depth_dir, f"/depth_{width}x{height}_{self.time_stamp}.png")
        cv2.imwrite(filename, data)
        print(colored(f"\nDepth image saved({data.shape}):\n{filename}", "yellow"))

        data = np.array(data)
        return data


    def _save_color_frame(self, frame: ColorFrame):
        """
        Save RGB image to file

        Args:
            frame (ColorFrame): RGB frame from get_color_frame()

        Returns:
            image (np.ndarray): Image matrix
        """
        # Process frame to bgr image
        image = frame_to_bgr_image(frame)
        if image is None:
            raise RuntimeError(colored("\nfailed to convert frame to image", "red"))
        
        # Image infomation
        width = frame.get_width()
        height = frame.get_height()

        # Create RGB image save directory
        save_image_dir = os.path.join(self.save_path, "RGB_images")
        os.makedirs(save_image_dir, exist_ok=True)
        # Set filename
        filename = os.path.join(save_image_dir, f"/RGB_{width}x{height}_{self.time_stamp}.png")
        # Save image
        cv2.imwrite(filename, image)
        print(colored(f"\nRGB image saved({image.shape}):\n{filename}", "yellow"))

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        return image
    
    def _save_point_cloud(self, frame):
        # Config point cloud filter
        point_format = OBFormat.RGB_POINT
        self.point_cloud_filter.set_create_point_format(point_format)
        # Process frame using configed filter
        point_cloud_frame = self.point_cloud_filter.process(frame)

        # Create point clouds save directory
        save_pointcloud_dir = os.path.join(self.save_path, "point_clouds")
        os.makedirs(save_pointcloud_dir, exist_ok=True)
        # Save point cloud ply file
        filename = os.path.join(save_pointcloud_dir, f"point_cloud_{self.time_stamp}.ply")
        save_point_cloud_to_ply(filename, point_cloud_frame)

        # Get RGBD point cloud data
        point_cloud = self.point_cloud_filter.calculate(point_cloud_frame)

        return point_cloud
    
    def _check_pointcloud(self, frame):
        point_format = OBFormat.RGB_POINT
        self.point_cloud_filter.set_create_point_format(point_format)

        point_cloud_frame = self.point_cloud_filter.process(frame)
        if point_cloud_frame is None:
            return False
        else:
            return True
        
    def _check_depth(self, depth_frame):
        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()
        depth_data_size = depth_frame.get_data_size()
        if depth_data_size != depth_width * depth_height * 2:
            return False
        else:
            return True
    
    def _depth_post_process(self, depth_frame):
        """
        Post-Process the depth data using recommended filter

        Args:
            depth_frame (np.ndarray): Depth frame from get_depth_frame()

        Returns:
            depth_frame (np.ndarray): Processed depth frame
        """
        for i in range(len(self.filter_list)):
            post_filter = self.filter_list[i]
            if post_filter and post_filter.is_enabled() and depth_frame:
                depth_data_size = depth_frame.get_data()
                if len(depth_data_size) < (depth_frame.get_width() * depth_frame.get_height() * 2):
                    raise RuntimeError(colored("depth data is not complete", "red"))

                new_depth_frame = post_filter.process(depth_frame)
                depth_frame = new_depth_frame.as_depth_frame() 

    
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
    
    def get_point_xyz_from_pointcloud(
        self,
        points: np.ndarray,
        u: int,
        v: int
    ) -> Tuple[float, float, float]:
        """
        Given a flattened RGBD point‐cloud array and a pixel (u, v) in the aligned RGB image,
        return the (X, Y, Z) in camera coordinates.

        Args:
            points: np.ndarray of shape (H*W, 6), each row = (X, Y, Z, R, G, B)
            u (int): pixel column, 0 <= u < self.RGB_width
            v (int): pixel row,    0 <= v < self.RGB_height

        Returns:
            (X, Y, Z): float tuple in mm.
        """
        H = self.RGB_height
        W = self.RGB_width

        if points.ndim != 2 or points.shape[0] != H * W or points.shape[1] < 3:
            raise ValueError(colored(f"points must be (H*W, >=3), got {points.shape}", "red"))

        if not (0 <= u < W and 0 <= v < H):
            raise ValueError(colored(f"pixel ({u},{v}) out of bounds 0–{W-1},0–{H-1}", "red"))

        idx = v * W + u
        X, Y, Z = points[idx, 0], points[idx, 1], points[idx, 2]
        return X, Y, Z
    
       
    # =================================Camera Action=================================
    def take_photo(self):
        """
        Capture one RGB+Depth frame from the Gemini 355 camera.

        Args:
            None

        Returns:
            Tuple: 
                - time_stamp: int
                - RGB_image: BGR color image (png) as a numpy.ndarray.
                - Depth_graph: Depth Graph (png) as a numpy.ndarray.
                - Point: point clouds as a numpy.ndarray
        """
        self.time_stamp = int(time.time() * 1000)
        frames = None

        while True:
            frames = self.pipeline.wait_for_frames(100)
            # Check frames
            if frames is None:
                continue
            # Check point cloud data
            frame = self.align_filter.process(frames)
            point_cloud_tag = self._check_pointcloud(frame)
            if point_cloud_tag == False:
                continue
            # Check RGB and Depth data
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame == None or depth_frame == None:
                continue
            # Align RGB and depth
            frames = self.align_filter.process(frames)
            if not frames:
                continue
            frames = frames.as_frame_set()
            # Check RGB and Depth data
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame == None or depth_frame == None:
                continue
            # Check Depth
            depth_tag = self._check_depth(depth_frame)
            if depth_tag == False:
                continue
            else:
                break
        
        # Save point cloud
        point_cloud = self._save_point_cloud(frame)
        # Save RGB image
        RGB_image = self._save_color_frame(color_frame)
        # Post-Process and Save Depth Image
        depth_frame = self._depth_post_process(depth_frame)
        depth_graph = self._save_depth_frame(depth_frame)
        print()

        return (self.time_stamp, RGB_image, depth_graph, point_cloud)
