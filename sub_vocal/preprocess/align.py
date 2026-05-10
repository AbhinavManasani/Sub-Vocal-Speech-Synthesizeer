"""
Lip Region Alignment
====================

Performs affine alignment of lip crops to a canonical 96×96 reference frame.
This normalizes rotation, scale, and position so that AV-HuBERT receives
consistent input regardless of head pose in the original video.

Reference: AV-HuBERT preparation pipeline uses similar affine normalization
based on face landmarks to center the mouth region.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

# Canonical reference points for a centered mouth in a 96x96 image
# These are approximate positions for left corner, right corner, and bottom lip center
CANONICAL_LIP_POINTS = np.array(
    [
        [24.0, 48.0],   # Left mouth corner
        [72.0, 48.0],   # Right mouth corner
        [48.0, 72.0],   # Bottom lip center
    ],
    dtype=np.float32,
)

# MediaPipe indices for alignment anchors
# 61 = left mouth corner, 291 = right mouth corner, 17 = bottom lip center
ALIGNMENT_ANCHOR_INDICES = [61, 291, 17]

# Additional face anchors for more stable alignment
# Left eye outer, right eye outer, nose tip
FACE_ANCHOR_INDICES = [33, 263, 1]


class LipAligner:
    """
    Aligns lip crops to a canonical 96×96 reference frame using affine transforms.
    
    The aligner uses three anchor points (left corner, right corner, bottom center)
    to compute a similarity transform that normalizes rotation and scale.
    
    Usage:
        aligner = LipAligner(target_size=(96, 96))
        aligned = aligner.align_frame(frame, detection)
        aligned_batch = aligner.align_video(frames, detections)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (96, 96),
        use_face_anchors: bool = False,
    ):
        """
        Initialize the lip aligner.
        
        Args:
            target_size: Output dimensions (width, height)
            use_face_anchors: If True, use eye+nose anchors instead of lip corners
                              (more stable but includes more face context)
        """
        self.target_size = target_size
        self.use_face_anchors = use_face_anchors

        # Scale canonical points to target size
        scale_x = target_size[0] / 96.0
        scale_y = target_size[1] / 96.0
        self.canonical_points = CANONICAL_LIP_POINTS.copy()
        self.canonical_points[:, 0] *= scale_x
        self.canonical_points[:, 1] *= scale_y

        if use_face_anchors:
            # Use eye-nose triangle for alignment (more stable)
            self.anchor_indices = FACE_ANCHOR_INDICES
            # Canonical positions for eyes + nose in target frame
            self.canonical_points = np.array(
                [
                    [target_size[0] * 0.3, target_size[1] * 0.25],  # Left eye
                    [target_size[0] * 0.7, target_size[1] * 0.25],  # Right eye
                    [target_size[0] * 0.5, target_size[1] * 0.55],  # Nose tip
                ],
                dtype=np.float32,
            )
        else:
            self.anchor_indices = ALIGNMENT_ANCHOR_INDICES

        logger.info(
            f"LipAligner initialized: target={target_size}, "
            f"anchors={'face' if use_face_anchors else 'lip'}"
        )

    def _get_anchor_points(self, detection: Dict) -> Optional[np.ndarray]:
        """
        Extract 3 anchor points from detection for affine estimation.
        
        Args:
            detection: Detection dict from LipDetector.detect_lips()
            
        Returns:
            3x2 array of anchor points in pixel coordinates, or None
        """
        if detection is None:
            return None

        all_landmarks = detection.get("all_face_landmarks")
        if all_landmarks is None:
            # Fallback: use lip landmarks directly
            lip_lm = detection["landmarks"]
            if len(lip_lm) < 3:
                return None
            # Use first, middle, and last lip points
            n = len(lip_lm)
            return np.array(
                [lip_lm[0], lip_lm[n // 2], lip_lm[-1]], dtype=np.float32
            )

        try:
            anchors = np.array(
                [all_landmarks[idx] for idx in self.anchor_indices],
                dtype=np.float32,
            )
            return anchors
        except (IndexError, KeyError):
            return None

    def compute_affine_matrix(
        self, src_points: np.ndarray
    ) -> np.ndarray:
        """
        Compute the 2x3 affine transformation matrix from source to canonical.
        
        Uses cv2.getAffineTransform for an exact 3-point mapping.
        
        Args:
            src_points: 3x2 array of source anchor points
            
        Returns:
            2x3 affine transformation matrix
        """
        M = cv2.getAffineTransform(src_points, self.canonical_points)
        return M

    def compute_similarity_transform(
        self, src_points: np.ndarray
    ) -> np.ndarray:
        """
        Compute a similarity transform (rotation + uniform scale + translation)
        using least-squares fitting. More robust than exact affine.
        
        Args:
            src_points: Nx2 array of source points (N >= 2)
            
        Returns:
            2x3 affine matrix representing the similarity transform
        """
        # Use OpenCV's estimateAffinePartial2D for similarity transform
        M, _ = cv2.estimateAffinePartial2D(
            src_points.reshape(-1, 1, 2),
            self.canonical_points.reshape(-1, 1, 2),
        )
        if M is None:
            # Fallback to identity with centering
            M = np.eye(2, 3, dtype=np.float64)
        return M

    def align_frame(
        self,
        frame: np.ndarray,
        detection: Optional[Dict],
        grayscale: bool = True,
        method: str = "affine",
    ) -> np.ndarray:
        """
        Align a single frame to the canonical lip position.
        
        Args:
            frame: Input BGR image (H, W, 3)
            detection: Detection dict from LipDetector, or None
            grayscale: Convert result to grayscale
            method: 'affine' (exact 3-point) or 'similarity' (robust)
            
        Returns:
            Aligned lip crop as numpy array [target_h, target_w] or [target_h, target_w, 3]
        """
        w, h = self.target_size

        if detection is None:
            logger.debug("No detection, returning zero frame")
            return np.zeros((h, w), dtype=np.uint8) if grayscale else np.zeros(
                (h, w, 3), dtype=np.uint8
            )

        src_points = self._get_anchor_points(detection)
        if src_points is None:
            logger.debug("Cannot extract anchors, returning center crop")
            return self._center_crop(frame, grayscale)

        # Compute transform
        if method == "similarity":
            M = self.compute_similarity_transform(src_points)
        else:
            M = self.compute_affine_matrix(src_points)

        # Apply warp
        aligned = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if grayscale and len(aligned.shape) == 3:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        return aligned

    def _center_crop(self, frame: np.ndarray, grayscale: bool) -> np.ndarray:
        """Fallback: take a center crop from the frame."""
        h, w_frame = frame.shape[:2]
        target_w, target_h = self.target_size
        crop_size = min(h, w_frame, max(target_w, target_h) * 2)

        y1 = max(0, (h - crop_size) // 2)
        x1 = max(0, (w_frame - crop_size) // 2)
        crop = frame[y1 : y1 + crop_size, x1 : x1 + crop_size]
        crop = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_CUBIC)

        if grayscale and len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        return crop

    def align_video(
        self,
        frames: np.ndarray,
        detections: List[Optional[Dict]],
        raw_frames: Optional[List[np.ndarray]] = None,
        grayscale: bool = True,
        method: str = "affine",
    ) -> np.ndarray:
        """
        Align all frames from a video to the canonical lip position.
        
        This is the main entry point for pipeline integration.
        If raw_frames is provided, alignment is done from the original video frames.
        Otherwise, it aligns the already-cropped lip regions.
        
        Args:
            frames: Pre-cropped lip frames [T, H, W] or [T, H, W, 3]
            detections: List of detection dicts per frame
            raw_frames: Optional original video frames for direct alignment
            grayscale: Convert results to grayscale
            method: 'affine' or 'similarity'
            
        Returns:
            Aligned frames as numpy array [T, target_h, target_w]
        """
        aligned_frames = []
        smoothed_detections = self._smooth_detections(detections)

        for i in range(len(frames)):
            if raw_frames is not None:
                source_frame = raw_frames[i]
            else:
                source_frame = frames[i]
                if len(source_frame.shape) == 2:
                    source_frame = cv2.cvtColor(source_frame, cv2.COLOR_GRAY2BGR)

            det = smoothed_detections[i] if i < len(smoothed_detections) else None
            aligned = self.align_frame(source_frame, det, grayscale, method)
            aligned_frames.append(aligned)

        result = np.stack(aligned_frames, axis=0)
        logger.info(f"Aligned {len(aligned_frames)} frames, shape: {result.shape}")
        return result

    def _smooth_detections(
        self, detections: List[Optional[Dict]], window: int = 3
    ) -> List[Optional[Dict]]:
        """
        Smooth detection landmarks over time to reduce jitter.
        Uses a simple moving average over the landmark positions.
        
        Args:
            detections: List of per-frame detections
            window: Smoothing window size
            
        Returns:
            Smoothed detections list
        """
        if not detections or window <= 1:
            return detections

        smoothed = []
        for i in range(len(detections)):
            if detections[i] is None:
                smoothed.append(None)
                continue

            # Collect landmarks from neighboring frames
            neighbor_landmarks = []
            for j in range(max(0, i - window // 2), min(len(detections), i + window // 2 + 1)):
                if detections[j] is not None:
                    neighbor_landmarks.append(detections[j]["landmarks"])

            if not neighbor_landmarks:
                smoothed.append(detections[i])
                continue

            # Average the landmarks
            avg_landmarks = np.mean(neighbor_landmarks, axis=0)
            smoothed_det = detections[i].copy()
            smoothed_det["landmarks"] = avg_landmarks
            smoothed.append(smoothed_det)

        return smoothed
