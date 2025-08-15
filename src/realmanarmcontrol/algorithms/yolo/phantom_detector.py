#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhantomDet - AI-powered Phantom Detection System
Main inference program using trained YOLO weights
"""

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

# Add src directory to path for utility imports
sys.path.append(str(Path(__file__).parent))

try:
    from config import get_config
except ImportError:
    print("❌ Error: Cannot import configuration. Please ensure config.py exists.")
    sys.exit(1)


class PhantomDetector:
    """
    AI-powered phantom detection system using trained YOLO model
    """
    
    def __init__(self, weights_path: str = None, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        """
        Initialize PhantomDetector
        
        Args:
            weights_path: Path to trained YOLO weights (optional, uses best.pt by default)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.console = Console()
        self.config = get_config()
        
        # Model configuration
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Determine weights path
        if weights_path and Path(weights_path).exists():
            self.weights_path = Path(weights_path)
        else:
            # Use trained weights from config
            from config import BEST_MODEL_PATH
            if BEST_MODEL_PATH.exists():
                self.weights_path = BEST_MODEL_PATH
            else:
                raise FileNotFoundError(f"No trained weights found at {BEST_MODEL_PATH}. Please provide weights_path or ensure training results exist.")
        
        # Initialize model
        self.model = None
        self.class_names = ['phantom']  # Default class name
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.load_model()
        
        self.console.print(f"[green]✅ PhantomDetector initialized successfully![/green]")
        self.console.print(f"[cyan]Weights: {self.weights_path}[/cyan]")
        self.console.print(f"[cyan]Confidence: {self.conf_threshold}, IoU: {self.iou_threshold}[/cyan]")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'phantom_detector.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load trained YOLO model with error handling"""
        try:
            self.console.print("[cyan]Loading trained YOLO model...[/cyan]")
            self.model = YOLO(str(self.weights_path))
            
            # Get class names if available
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            
            self.logger.info(f"Model loaded successfully: {self.weights_path}")
            self.logger.info(f"Class names: {self.class_names}")
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to load model: {e}[/red]")
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def detect_single_image(self, image: Union[str, Path]) -> Dict:
        """
        Detect phantoms in a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Detection results dictionary
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = self.model(
            str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Process results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'class_name': self.class_names[int(box.cls[0].cpu().numpy())] if int(box.cls[0].cpu().numpy()) < len(self.class_names) else 'unknown'
                }
                detections.append(detection)
        
        return {
            'image_path': str(image_path),
            'image_shape': image.shape,
            'detections': detections,
            'detection_count': len(detections),
            'timestamp': datetime.now().isoformat()
        }
    
    def detect_batch(self, input_path: Union[str, Path], output_dir: Union[str, Path] = None) -> List[Dict]:
        """
        Detect phantoms in multiple images
        
        Args:
            input_path: Path to directory containing images
            output_dir: Directory to save results (optional)
            
        Returns:
            List of detection results
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if input_path.is_file():
            image_files = [input_path] if input_path.suffix.lower() in image_extensions else []
        else:
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f'*{ext}'))
                image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No valid image files found in: {input_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Path(__file__).parent / "results" / f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images with progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(f"[cyan]Detecting phantoms...", total=len(image_files))
            
            for image_file in image_files:
                try:
                    # Detect
                    result = self.detect_single_image(image_file)
                    results.append(result)
                    
                    # Save visualized result
                    if result['detection_count'] > 0:
                        self.save_annotated_image(image_file, result, output_dir)
                        # Also generate points visualization
                        self.points_to_reach(image_file, result, output_dir)
                    
                    progress.advance(task)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_file}: {e}")
                    progress.advance(task)
                    continue
        
        # Save batch results
        self.save_batch_results(results, output_dir)
        
        self.console.print(f"[green]✅ Batch detection completed![/green]")
        self.console.print(f"[cyan]Results saved to: {output_dir}[/cyan]")
        
        return results
    

    def points_to_reach(self, image_path: Path, result: Dict, output_dir: Path):
        """
        Find three equidistant points on the longer edge of the detection box
        and draw them on the image
        
        Args:
            image_path: Path to the input image
            result: Detection result dictionary
            output_dir: Directory to save the annotated image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return
        
        for detection in result['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw the detection box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Always place three points along the horizontal center line inside the box
            width = x2 - x1
            height = y2 - y1
            
            # Use horizontal center line (y = y1 + height/2)
            center_y = y1 + height // 2  # Horizontal center line
            # Three equidistant points along the horizontal center line
            p1x = x1 + width // 4
            p1y = center_y
            p2x = x1 + width // 2
            p2y = center_y
            p3x = x1 + 3 * width // 4
            p3y = center_y
            
            # Draw the three points on the image
            # Point 1 (red)
            cv2.circle(image, (p1x, p1y), 5, (0, 0, 255), -1)
            cv2.putText(image, "P1", (p1x + 10, p1y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Point 2 (blue)
            cv2.circle(image, (p2x, p2y), 5, (255, 0, 0), -1)
            cv2.putText(image, "P2", (p2x + 10, p2y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Point 3 (green)
            cv2.circle(image, (p3x, p3y), 5, (0, 255, 0), -1)
            cv2.putText(image, "P3", (p3x + 10, p3y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw horizontal center line to show where points are placed
            cv2.line(image, (x1, center_y), (x2, center_y), (255, 255, 0), 2)
        
        # Save the annotated image
        output_file = output_dir / f"points_{image_path.name}"
        cv2.imwrite(str(output_file), image)
        
        # Calculate and save points data (save as .npy array)
        points = []
        for detection in result['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            width = x2 - x1
            height = y2 - y1
            
            # Always place three points along the horizontal center line
            center_y = y1 + height // 2  # Horizontal center line
            p1 = (x1 + width // 4, center_y)
            p2 = (x1 + width // 2, center_y)
            p3 = (x1 + 3 * width // 4, center_y)
            
            points.append({
                'bbox': [x1, y1, x2, y2],
                'points': [p1, p2, p3],
                'line_type': 'horizontal_center',
                'width': width,
                'height': height
            })

        # Save as numpy array: shape (num_detections, 3, 2)
        if points:
            pts_array = np.array([[p for p in item['points']] for item in points], dtype=np.int32)
            npy_file = output_dir / f"points_{image_path.stem}.npy"
            np.save(str(npy_file), pts_array)
        
        return points
    
    def get_points_for_detection(self, result: Dict) -> List[Dict]:
        """
        Get the three equidistant points for a detection result without saving images
        
        Args:
            result: Detection result dictionary
            
        Returns:
            List of dictionaries containing bbox and points information
        """
        points = []
        for detection in result['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            width = x2 - x1
            height = y2 - y1
            
            # Always place three points along the horizontal center line
            center_y = y1 + height // 2  # Horizontal center line
            p1 = (x1 + width // 4, center_y)
            p2 = (x1 + width // 2, center_y)
            p3 = (x1 + 3 * width // 4, center_y)
            
            points.append({
                'bbox': [x1, y1, x2, y2],
                'points': [p1, p2, p3],
                'line_type': 'horizontal_center',
                'width': width,
                'height': height
            })
        
        return points

    def save_annotated_image(self, image_path: Path, result: Dict, output_dir: Path):
        """Save image with detection annotations"""
        image = cv2.imread(str(image_path))
        if image is None:
            return
        
        # Draw bounding boxes
        for detection in result['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save annotated image
        output_file = output_dir / f"annotated_{image_path.name}"
        cv2.imwrite(str(output_file), image)
    
    def save_batch_results(self, results: List[Dict], output_dir: Path):
        """Save batch detection results"""
        # Save JSON results
        json_file = output_dir / "detection_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_file = output_dir / "detection_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PhantomDet Detection Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total images processed: {len(results)}\n")
            
            total_detections = sum(r['detection_count'] for r in results)
            images_with_detections = sum(1 for r in results if r['detection_count'] > 0)
            
            f.write(f"Images with detections: {images_with_detections}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Detection rate: {images_with_detections/len(results)*100:.1f}%\n")
            f.write("\nDetailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                f.write(f"File: {Path(result['image_path']).name}\n")
                f.write(f"  Detections: {result['detection_count']}\n")
                if result['detections']:
                    for i, det in enumerate(result['detections']):
                        f.write(f"    {i+1}: {det['class_name']} (conf: {det['confidence']:.2f})\n")
                f.write("\n")
    
    def print_results_table(self, results: List[Dict]):
        """Print detection results in a formatted table"""
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return
        
        table = Table(title="Phantom Detection Results")
        table.add_column("Image", style="cyan")
        table.add_column("Detections", justify="center", style="magenta")
        table.add_column("Max Confidence", justify="center", style="green")
        table.add_column("Status", justify="center")
        
        for result in results:
            image_name = Path(result['image_path']).name
            count = result['detection_count']
            
            if count > 0:
                max_conf = max(det['confidence'] for det in result['detections'])
                status = f"[green]✓ {count} phantom(s)[/green]"
                conf_str = f"{max_conf:.2f}"
            else:
                status = "[dim]No detections[/dim]"
                conf_str = "-"
            
            table.add_row(image_name, str(count), conf_str, status)
        
        self.console.print(table)

    def get_detect_points(self, image: np.ndarray, model_path) -> List[Tuple[float]]:
        pass

    
def create_arg_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PhantomDet - AI-powered Phantom Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect in a single image
  python phantom_detector.py --input image.jpg
  
  # Batch detection in a directory
  python phantom_detector.py --input /path/to/images/ --output /path/to/results/
  
  # Use custom weights and thresholds
  python phantom_detector.py --input image.jpg --weights custom_weights.pt --conf 0.7 --iou 0.5
        """
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='Input image file or directory path')
    parser.add_argument('--output', '-o', 
                       help='Output directory for results (default: auto-generated)')
    parser.add_argument('--weights', '-w', 
                       help='Path to YOLO weights file (default: trained best.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save annotated images')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    
    return parser


def main():
    """Main function with comprehensive error handling"""
    console = Console()
    
    # Print banner
    banner = Panel.fit(
        "[bold blue]🔍 PhantomDet - AI Phantom Detection System[/bold blue]\n"
        "[dim]Powered by YOLOv11 + Custom Training[/dim]",
        border_style="blue"
    )
    console.print(banner)
    
    try:
        # Parse arguments
        parser = create_arg_parser()
        args = parser.parse_args()
        
        # Initialize detector
        detector = PhantomDetector(
            weights_path=args.weights,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        # Run detection
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image detection
            console.print(f"[cyan]Detecting phantoms in: {input_path.name}[/cyan]")
            result = detector.detect_single_image(input_path)
            
            # Print results
            if result['detection_count'] > 0:
                console.print(f"[green]✅ Found {result['detection_count']} phantom(s)![/green]")
                for i, det in enumerate(result['detections']):
                    console.print(f"  Detection {i+1}: {det['class_name']} (confidence: {det['confidence']:.2f})")
            else:
                console.print("[yellow]No phantoms detected[/yellow]")
            
            # Save annotated image if requested
            if not args.no_save:
                output_dir = Path(args.output) if args.output else Path("results") / "single_detection"
                output_dir.mkdir(parents=True, exist_ok=True)
                detector.save_annotated_image(input_path, result, output_dir)
                console.print(f"[cyan]Annotated image saved to: {output_dir}[/cyan]")
                
                # Also generate points visualization
                points = detector.points_to_reach(input_path, result, output_dir)
                if points:
                    console.print(f"[cyan]Points visualization saved to: {output_dir}[/cyan]")
                    console.print(f"[cyan]Found {len(points)} detection(s) with points[/cyan]")
        
        else:
            # Batch detection
            console.print(f"[cyan]Running batch detection in: {input_path}[/cyan]")
            results = detector.detect_batch(input_path, args.output)
            
            # Print summary table
            if not args.quiet:
                detector.print_results_table(results)
            
            # Print summary statistics
            total_detections = sum(r['detection_count'] for r in results)
            images_with_detections = sum(1 for r in results if r['detection_count'] > 0)
            
            console.print(f"\n[bold]Detection Summary:[/bold]")
            console.print(f"[green]Images processed: {len(results)}[/green]")
            console.print(f"[green]Images with detections: {images_with_detections}[/green]")
            console.print(f"[green]Total detections: {total_detections}[/green]")
            console.print(f"[green]Detection rate: {images_with_detections/len(results)*100:.1f}%[/green]")
        
        console.print("\n[bold green]🎉 Detection completed successfully![/bold green]")
        return True
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Detection interrupted by user[/yellow]")
        return False
        
    except Exception as e:
        console.print(f"\n[red]❌ Detection failed: {e}[/red]")
        logging.getLogger(__name__).error(f"Detection failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)