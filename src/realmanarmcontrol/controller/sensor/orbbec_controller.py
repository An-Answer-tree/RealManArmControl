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
    def _save_depth_frame(self, frame: "DepthFrame") -> np.ndarray:
        """Save a depth frame to a 16-bit PNG and return the raw depth array.

        Reads the device-provided depth buffer (millimeters, uint16), reshapes it to
        (H, W), writes it to disk, and returns the numpy array.

        Args:
            frame: Depth frame as returned by ``get_depth_frame()``.

        Returns:
            np.ndarray: Depth image of shape (H, W) with dtype ``np.uint16`` (mm).

        Raises:
            ValueError: If the frame payload size does not match ``H*W*2`` bytes.
            OSError: If writing the PNG fails.
        """
        width = self.Depth_width
        height = self.Depth_height

        # Sanity check on payload size
        payload = frame.get_data()
        if len(payload) != width * height * 2:
            raise ValueError(
                f"Depth payload size mismatch: got {len(payload)} bytes, "
                f"expected {width*height*2}."
            )

        # Convert to (H, W) uint16 depth map
        data = np.frombuffer(payload, dtype=np.uint16).reshape((height, width))

        # Ensure output directory exists
        save_depth_dir = os.path.join(self.save_path, "depth_images")
        os.makedirs(save_depth_dir, exist_ok=True)

        # Build filename (no leading slash!)
        filename = os.path.join(
            save_depth_dir, f"depth_{width}x{height}_{self.time_stamp}.png"
        )

        # Write 16-bit PNG
        if not cv2.imwrite(filename, data):
            raise OSError(f"Failed to write depth image: {filename}")

        print(colored(f"\nDepth image saved({data.shape}):\n{filename}", "yellow"))
        return data

    def _save_color_frame(self, frame: "ColorFrame") -> np.ndarray:
        """Save an RGB color frame to disk and return it as a NumPy array.

        Converts the incoming SDK frame to a BGR OpenCV image, persists it as PNG,
        and returns the image in **RGB** channel order.

        Args:
            frame: Color frame as returned by ``get_color_frame()``.

        Returns:
            np.ndarray: RGB image of shape (H, W, 3) with dtype ``np.uint8``.

        Raises:
            RuntimeError: If converting the input frame to an image fails.
            OSError: If writing the PNG fails.
        """
        # Convert SDK frame → BGR ndarray (OpenCV)
        image_bgr = frame_to_bgr_image(frame)
        if image_bgr is None:
            raise RuntimeError(colored("\nFailed to convert frame to image", "red"))

        width = self.RGB_width
        height = self.RGB_height

        # Prepare output directory
        save_image_dir = os.path.join(self.save_path, "RGB_images")
        os.makedirs(save_image_dir, exist_ok=True)

        # Build filename (no leading slash!)
        filename = os.path.join(
            save_image_dir, f"RGB_{width}x{height}_{self.time_stamp}.png"
        )

        # Persist BGR image
        if not cv2.imwrite(filename, image_bgr):
            raise OSError(f"Failed to write RGB image: {filename}")

        print(colored(f"\nRGB image saved({image_bgr.shape}):\n{filename}", "yellow"))

        # Return as RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return np.array(image_rgb)

    def _save_point_cloud(self, frame: "DepthFrame") -> np.ndarray:
        """Compute and save an RGB point cloud (.ply), returning the point data.

        Configures the point cloud filter to produce RGB points, processes the input
        frame, writes a PLY file, and returns the computed XYZRGB array.

        Args:
            frame: Input frame compatible with the configured point cloud filter
                (e.g., a synchronized depth/color frame or frameset).

        Returns:
            np.ndarray: Array of shape (N, 6) or (H*W, 6) with columns
                ``[x, y, z, r, g, b]``. The exact shape depends on the SDK output.

        Raises:
            OSError: If saving the PLY file fails.
            RuntimeError: If point cloud calculation fails.
        """
        point_format = OBFormat.RGB_POINT
        self.point_cloud_filter.set_create_point_format(point_format)

        point_cloud_frame = self.point_cloud_filter.process(frame)
        if point_cloud_frame is None:
            raise RuntimeError("Point cloud filter returned None.")

        save_pointcloud_dir = os.path.join(self.save_path, "point_clouds")
        os.makedirs(save_pointcloud_dir, exist_ok=True)

        filename = os.path.join(save_pointcloud_dir, f"point_cloud_{self.time_stamp}.ply")
        save_point_cloud_to_ply(filename, point_cloud_frame)

        # Calculate XYZRGB ndarray from the frame
        point_cloud = self.point_cloud_filter.calculate(point_cloud_frame)
        if point_cloud is None:
            raise RuntimeError("Failed to calculate point cloud data.")

        return point_cloud

    def _check_pointcloud(self, frame: "DepthFrame") -> bool:
        """Check whether a point cloud frame can be produced for the given input.

        Args:
            frame: Input frame to feed to the point cloud filter.

        Returns:
            bool: ``True`` if the filter produces a non-None point cloud frame;
            ``False`` otherwise.
        """
        point_format = OBFormat.RGB_POINT
        self.point_cloud_filter.set_create_point_format(point_format)

        point_cloud_frame = self.point_cloud_filter.process(frame)
        return point_cloud_frame is not None

    def _check_depth(self, depth_frame: "DepthFrame") -> bool:
        """Validate the raw depth frame payload size against width/height.

        Args:
            depth_frame: Depth frame to validate.

        Returns:
            bool: ``True`` if ``data_size == width * height * 2``; ``False`` otherwise.
        """
        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()
        depth_data_size = depth_frame.get_data_size()
        return depth_data_size == depth_width * depth_height * 2

    def _depth_post_process(self, depth_frame: "DepthFrame") -> "DepthFrame":
        """Apply the configured post-filters to a depth frame.

        Iterates over ``self.filter_list`` and applies each enabled filter in order.
        Performs a completeness check before filtering.

        Args:
            depth_frame: Input depth frame (uint16, mm) from ``get_depth_frame()``.

        Returns:
            DepthFrame: The filtered depth frame (SDK object, not a NumPy array).

        Raises:
            RuntimeError: If the input depth data is incomplete.
        """
        for post_filter in self.filter_list:
            if post_filter and post_filter.is_enabled() and depth_frame:
                payload = depth_frame.get_data()
                if len(payload) < (depth_frame.get_width() * depth_frame.get_height() * 2):
                    raise RuntimeError(colored("Depth data is not complete", "red"))

                new_depth_frame = post_filter.process(depth_frame)
                depth_frame = new_depth_frame.as_depth_frame()
        return depth_frame

    
    # =================================Algorithm=================================
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
        """Capture one RGB + Depth frame and persist outputs to disk.

        This method blocks until a valid, aligned frame set is obtained from the
        Gemini 355 camera. It performs basic sanity checks (frame presence, point
        cloud availability, depth payload size), then:
        1) Saves a colored point cloud (.ply).
        2) Saves an RGB image (PNG).
        3) Post-processes and saves a depth image (uint16 PNG, millimeters).

        Args:
            None.

        Returns:
            Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
                A 4-tuple ``(time_stamp, RGB_image, depth_graph, point_cloud)`` where:
                - ``time_stamp``: ``int``. Milliseconds since epoch at capture start.
                - ``RGB_image``: ``np.ndarray``. RGB image of shape ``(H, W, 3)``,
                dtype ``uint8`` (converted to RGB before return).
                - ``depth_graph``: ``np.ndarray``. Depth map of shape ``(H, W)``,
                dtype ``uint16`` in millimeters.
                - ``point_cloud``: ``np.ndarray``. SDK-dependent layout, typically
                ``(N, 6)`` or ``(H*W, 6)`` with columns ``[x, y, z, r, g, b]``.

        Notes:
            - The actual file I/O is delegated to helper methods:
            ``_save_point_cloud``, ``_save_color_frame``, and ``_save_depth_frame``.
            - ``self.align_filter`` is applied before point-cloud validation and again
            before saving color/depth to ensure alignment.
        """
        # Generate a millisecond timestamp for filenames and return payload.
        self.time_stamp = int(time.time() * 1000)

        # Initialize frame holder for the polling loop.
        frames = None

        # Poll the pipeline until a complete, valid, aligned frame set is obtained.
        while True:
            # Wait for a new frameset with a 100 ms timeout.
            frames = self.pipeline.wait_for_frames(100)

            # If nothing arrived within the timeout, try again.
            if frames is None:
                continue

            # Align frames (e.g., depth to color) and verify point-cloud feasibility.
            frame = self.align_filter.process(frames)
            point_cloud_tag = self._check_pointcloud(frame)
            if point_cloud_tag == False:
                # Point cloud not available yet; keep polling.
                continue

            # Basic presence check for color and depth frames before alignment.
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame == None or depth_frame == None:
                # Missing one of the streams; keep polling.
                continue

            # Align again to ensure RGB and depth share the same geometry.
            frames = self.align_filter.process(frames)
            if not frames:
                # Alignment produced no output; keep polling.
                continue
            frames = frames.as_frame_set()

            # Re-fetch aligned color and depth frames from the aligned frameset.
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame == None or depth_frame == None:
                # One of the aligned streams is missing; keep polling.
                continue

            # Validate depth payload size (width * height * 2 bytes for uint16).
            depth_tag = self._check_depth(depth_frame)
            if depth_tag == False:
                # Incomplete depth data; keep polling.
                continue
            else:
                # All checks passed; proceed to save and return.
                break

        # Save point cloud using the previously aligned frame.
        point_cloud = self._save_point_cloud(frame)

        # Save RGB image; helper returns an RGB ndarray.
        RGB_image = self._save_color_frame(color_frame)

        # Post-process depth (filters) and save as uint16 PNG; helper returns ndarray.
        depth_frame = self._depth_post_process(depth_frame)
        depth_graph = self._save_depth_frame(depth_frame)

        # Optional spacing in console output.
        print()

        # Return: timestamp, RGB image (RGB), depth map (uint16, mm), and point cloud.
        return (self.time_stamp, RGB_image, depth_graph, point_cloud)
