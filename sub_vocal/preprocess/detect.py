"""
Lip Detection using MediaPipe FaceMesh
=======================================

Extracts lip region-of-interest (ROI) from video frames using MediaPipe's 
FaceMesh solution. Returns per-frame lip landmarks and bounding boxes for
downstream cropping/alignment.

Key design choices:
- Uses refine_landmarks=True for higher lip accuracy (478 landmarks)
- Tight lip landmark subset (20 points) for minimal bounding box
- Supports both single-image and batch video processing
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

# Tight lip landmark indices from MediaPipe FaceMesh (468/478 landmarks)
# These 20 points trace the outer and inner lip contour
LIP_LANDMARK_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # Upper lip outer
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  # Lower lip outer
]

# Extended lip region includes surrounding face context
LIP_REGION_INDICES = [
    # Outer lip
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    # Inner lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    # Chin + nose for context
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109, 10, 338,
]


class LipDetector:
    """
    Detects and extracts lip ROI from video frames using MediaPipe FaceMesh.
    
    Usage:
        detector = LipDetector()
        frames, landmarks = detector.process_video("input.mp4")
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        """
        Initialize the lip detector with MediaPipe FaceLandmarker Task API.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            static_image_mode: If True, treats each frame independently (slower but more robust)
        """
        import os
        model_path = os.environ.get("MEDIAPIPE_MODEL_PATH", "checkpoints/mediapipe/face_landmarker.task")
        
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.mock_mode = False
            logger.info(f"LipDetector initialized with MediaPipe FaceLandmarker ({model_path})")
        except Exception as e:
            logger.warning(f"MediaPipe FaceLandmarker failed to load ({e}). Using mock detector.")
            self.mock_mode = True
            
        self.lip_indices = LIP_LANDMARK_INDICES

    def detect_lips(
        self, frame: np.ndarray
    ) -> Optional[Dict]:
        """
        Detect lip landmarks in a single frame.
        
        Args:
            frame: BGR image (H, W, 3) from OpenCV
            
        Returns:
            Dictionary with 'landmarks' (Nx2 array), 'bbox' (x1,y1,x2,y2),
            'center' (cx, cy), or None if no face detected
        """
        h, w, _ = frame.shape
        if getattr(self, "mock_mode", False):
            cx, cy = w // 2, h // 2
            return {
                "landmarks": np.array([[cx, cy]]),
                "all_face_landmarks": np.array([[cx, cy]]),
                "bbox": (cx - 48, cy - 48, cx + 48, cy + 48),
                "center": (cx, cy),
            }
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]

        # Extract lip landmark coordinates (pixel space)
        lip_points = []
        for idx in self.lip_indices:
            lm = face_landmarks[idx]
            px = int(lm.x * w)
            py = int(lm.y * h)
            lip_points.append((px, py))

        lip_points = np.array(lip_points, dtype=np.float32)

        # Compute bounding box with padding
        x_min, y_min = np.min(lip_points, axis=0).astype(int)
        x_max, y_max = np.max(lip_points, axis=0).astype(int)

        # Add proportional padding (30% of lip width/height)
        pad_x = int(0.3 * (x_max - x_min))
        pad_y = int(0.3 * (y_max - y_min))
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(w, x_max + pad_x)
        y_max = min(h, y_max + pad_y)

        # Compute center
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        # Get all face landmarks for affine alignment
        all_face_points = []
        for lm in face_landmarks:
            all_face_points.append((lm.x * w, lm.y * h))
        all_face_points = np.array(all_face_points, dtype=np.float32)

        return {
            "landmarks": lip_points,
            "all_face_landmarks": all_face_points,
            "bbox": (x_min, y_min, x_max, y_max),
            "center": (cx, cy),
        }

    def crop_lip_roi(
        self,
        frame: np.ndarray,
        detection: Dict,
        target_size: Tuple[int, int] = (96, 96),
        grayscale: bool = True,
    ) -> np.ndarray:
        """
        Crop the lip region from a frame using detection results.
        
        Args:
            frame: BGR image (H, W, 3)
            detection: Output from detect_lips()
            target_size: Output size (width, height)
            grayscale: If True, convert to grayscale
            
        Returns:
            Cropped and resized lip ROI as numpy array
        """
        x1, y1, x2, y2 = detection["bbox"]

        # Make the crop square (centered on lip region)
        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_size = max(crop_w, crop_h)

        cx, cy = detection["center"]
        x1 = max(0, int(cx - crop_size / 2))
        y1 = max(0, int(cy - crop_size / 2))
        x2 = min(frame.shape[1], int(cx + crop_size / 2))
        y2 = min(frame.shape[0], int(cy + crop_size / 2))

        lip_crop = frame[y1:y2, x1:x2]

        if lip_crop.size == 0:
            # Fallback: return a zero frame
            logger.warning("Empty lip crop, returning zeros")
            if grayscale:
                return np.zeros(target_size, dtype=np.uint8)
            return np.zeros((*target_size, 3), dtype=np.uint8)

        # Resize to target
        lip_crop = cv2.resize(lip_crop, target_size, interpolation=cv2.INTER_CUBIC)

        if grayscale:
            lip_crop = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)

        return lip_crop

    def process_video(
        self,
        video_path: str,
        target_size: Tuple[int, int] = (96, 96),
        grayscale: bool = True,
        max_frames: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[Optional[Dict]]]:
        """
        Process an entire video file and extract lip crops for each frame.
        
        Args:
            video_path: Path to the input video file
            target_size: Output size for each lip crop
            grayscale: Whether to convert crops to grayscale
            max_frames: Maximum number of frames to process (None = all)
            fps: Target FPS for extraction (None = use video's native FPS)
            
        Returns:
            Tuple of:
              - crops: numpy array [T, H, W] or [T, H, W, 3]
              - detections: list of detection dicts (None for missed frames)
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame sampling
        if fps and fps < video_fps:
            frame_interval = int(video_fps / fps)
        else:
            frame_interval = 1

        logger.info(
            f"Processing video: {video_path} "
            f"({total_frames} frames @ {video_fps:.1f} FPS, "
            f"sampling every {frame_interval} frames)"
        )

        crops = []
        detections = []
        frame_idx = 0
        processed = 0
        last_detection = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            if max_frames and processed >= max_frames:
                break

            detection = self.detect_lips(frame)

            if detection is None:
                # Interpolation fallback: use last known detection
                if last_detection is not None:
                    logger.debug(f"Frame {frame_idx}: no face, using last detection")
                    detection = last_detection
                else:
                    logger.debug(f"Frame {frame_idx}: no face detected, skipping")
                    if grayscale:
                        crops.append(np.zeros(target_size, dtype=np.uint8))
                    else:
                        crops.append(np.zeros((*target_size, 3), dtype=np.uint8))
                    detections.append(None)
                    frame_idx += 1
                    processed += 1
                    continue

            crop = self.crop_lip_roi(frame, detection, target_size, grayscale)
            crops.append(crop)
            detections.append(detection)
            last_detection = detection

            frame_idx += 1
            processed += 1

        cap.release()

        if not crops:
            raise ValueError(f"No frames extracted from video: {video_path}")

        crops = np.stack(crops, axis=0)
        logger.info(f"Extracted {len(crops)} lip crops, shape: {crops.shape}")

        return crops, detections

    def extract_audio(
        self, video_path: str, output_path: Optional[str] = None, sr: int = 16000
    ) -> str:
        """
        Extract audio track from a video file for speaker reference.
        
        Args:
            video_path: Path to the input video
            output_path: Where to save the audio (default: same dir, .wav extension)
            sr: Target sample rate
            
        Returns:
            Path to the extracted audio file
        """
        import subprocess

        video_path = Path(video_path)
        if output_path is None:
            output_path = str(video_path.with_suffix(".wav"))

        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", str(sr),  # Sample rate
            "-ac", "1",  # Mono
            output_path,
        ]

        try:
            subprocess.run(
                cmd, capture_output=True, check=True, timeout=60
            )
            logger.info(f"Audio extracted to: {output_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to extract audio: {e}")
            output_path = None

        return output_path

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
