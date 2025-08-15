import os
import sys
import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from ultralytics import YOLO


class Detector:
    def __init__(
        self, 
        weights_path: str, 
        conf_threshold: float = 0.5, 
        iou_threshold: float = 0.45,
    ):
        """
        Initialize PhantomDetector
        
        Args:
            weights_path: Path to trained YOLO weights (optional, uses best.pt by default)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.console = Console()
        
        # Model configuration
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.weights_path = Path(weights_path)
        
        # Initialize model
        self.model = None
        self.class_names = ['phantom']  # Default class name

        
        # Load model
        self.load_model()
        
        self.console.print(f"[green]✅ PhantomDetector initialized successfully![/green]")
        self.console.print(f"[cyan]Weights: {self.weights_path}[/cyan]")
        self.console.print(f"[cyan]Confidence: {self.conf_threshold}, IoU: {self.iou_threshold}[/cyan]")

    
    def load_model(self):
        """Load trained YOLO model with error handling"""
        try:
            self.console.print("[cyan]Loading trained YOLO model...[/cyan]")
            self.model = YOLO(str(self.weights_path))
            
            # Get class names if available
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())

        except Exception as e:
            self.console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise

    def get_detect_points(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Single-target detection: return three equally spaced points along the
        horizontal centerline of the detected bounding box.
        
        Args:
            image: numpy ndarray in BGR format
        
        Returns:
            List of three points [(x1, y1), (x2, y2), (x3, y3)]; empty list if no detection
        """
        try:
            self.console.print("[cyan]Detecting phantom in image...[/cyan]")
            
            # keep only a single detection
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=1,
                verbose=False
            )
            
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                self.console.print("[yellow]No detections found.[/yellow]")
                return []
            
            # take the first bounding box (x1, y1, x2, y2)
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            
            width = x2i - x1i
            height = y2i - y1i
            center_y = y1i + height // 2
            p1 = (x1i + width // 4, center_y)
            p2 = (x1i + width // 2, center_y)
            p3 = (x1i + 3 * width // 4, center_y)
            
            points = [p1, p2, p3]
            self.console.print(f"[green]✅ Points: {points}[/green]")
            return points
        except Exception as e:
            self.console.print(f"[red]❌ Detection failed: {e}[/red]")
            raise