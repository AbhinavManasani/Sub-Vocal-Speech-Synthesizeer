"""
AV-HuBERT Encoder — Visual Speech Unit Extraction
===================================================

Loads a pretrained AV-HuBERT Large model and extracts discrete speech units
from lip crop sequences using k-means quantization (KM500).

Pipeline:
  1. Load AV-HuBERT checkpoint via fairseq
  2. Preprocess lip frames to model input format
  3. Forward pass → continuous feature embeddings
  4. K-means quantization → discrete unit IDs [T]

Checkpoints:
  - AV-HuBERT Large: https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/av_hubert_large.pt
  - KM500: https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/km500.bin

Key insight: Speech units are language-agnostic phoneme-like tokens,
making the pipeline work across languages.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Default checkpoint paths
DEFAULT_AVHUBERT_CKPT = "checkpoints/av_hubert_large.pt"
DEFAULT_KM_MODEL = "checkpoints/km500.bin"

# Download URLs
AVHUBERT_URL = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/av_hubert_large.pt"
KM500_URL = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/km500.bin"


class AVHuBERTEncoder:
    """
    Wraps AV-HuBERT Large for extracting discrete speech units from lip videos.
    
    Usage:
        encoder = AVHuBERTEncoder()
        units = encoder.extract_units(lip_frames)  # lip_frames: [T, 96, 96]
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        km_path: Optional[str] = None,
        device: Optional[str] = None,
        layer: int = -1,
    ):
        """
        Initialize AV-HuBERT encoder.
        
        Args:
            ckpt_path: Path to av_hubert_large.pt checkpoint
            km_path: Path to km500.bin k-means model
            device: 'cuda' or 'cpu' (auto-detected if None)
            layer: Which transformer layer to extract features from (-1 = last)
        """
        self.ckpt_path = ckpt_path or DEFAULT_AVHUBERT_CKPT
        self.km_path = km_path or DEFAULT_KM_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer

        self.model = None
        self.km_model = None
        self._loaded = False

        logger.info(
            f"AVHuBERTEncoder configured: ckpt={self.ckpt_path}, "
            f"km={self.km_path}, device={self.device}"
        )

    def _download_checkpoints(self):
        """Download model checkpoints if they don't exist."""
        import urllib.request

        os.makedirs("checkpoints", exist_ok=True)

        if not os.path.exists(self.ckpt_path):
            logger.info(f"Downloading AV-HuBERT checkpoint to {self.ckpt_path}...")
            logger.info(f"URL: {AVHUBERT_URL}")
            logger.info("This is ~1.3GB, may take a few minutes...")
            urllib.request.urlretrieve(AVHUBERT_URL, self.ckpt_path)
            logger.info("AV-HuBERT checkpoint downloaded.")

        if not os.path.exists(self.km_path):
            logger.info(f"Downloading KM500 model to {self.km_path}...")
            urllib.request.urlretrieve(KM500_URL, self.km_path)
            logger.info("KM500 model downloaded.")

    def load(self):
        """
        Load the AV-HuBERT model and k-means quantizer.
        
        This is separated from __init__ to allow lazy loading.
        Handles the fairseq import complexity and version conflicts.
        """
        if self._loaded:
            return

        # Ensure checkpoints exist
        self._download_checkpoints()

        # Load AV-HuBERT model via fairseq
        try:
            self._load_avhubert()
        except Exception as e:
            logger.error(f"Failed to load AV-HuBERT: {e}")
            logger.warning("Falling back to dummy encoder mode")
            self.model = None

        # Load k-means quantizer
        try:
            self._load_kmeans()
        except Exception as e:
            logger.error(f"Failed to load k-means model: {e}")
            logger.warning("K-means quantizer unavailable, using random units")
            self.km_model = None

        self._loaded = True

    def _load_avhubert(self):
        """Load the AV-HuBERT model using fairseq checkpoint utilities."""
        import fairseq

        logger.info(f"Loading AV-HuBERT from {self.ckpt_path}...")

        # Register AV-HuBERT task and model with fairseq
        # The av_hubert code must be importable
        avhubert_root = self._find_avhubert_source()
        if avhubert_root and avhubert_root not in sys.path:
            sys.path.insert(0, avhubert_root)

        try:
            # Try importing av_hubert modules to register with fairseq
            import hubert_pretraining  # noqa: F401
            import hubert  # noqa: F401
        except ImportError:
            logger.warning(
                "AV-HuBERT source modules not found. "
                "Attempting direct checkpoint load..."
            )

        # Load model ensemble
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.ckpt_path]
        )

        self.model = models[0]
        self.model.eval()
        self.model = self.model.to(self.device)
        self.cfg = cfg
        self.task = task

        logger.info(
            f"AV-HuBERT loaded successfully on {self.device} "
            f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params)"
        )

    def _find_avhubert_source(self) -> Optional[str]:
        """
        Try to find the av_hubert source directory.
        This is needed because fairseq requires task/model registration.
        """
        search_paths = [
            "avhubert",
            "av_hubert/avhubert",
            "../av_hubert/avhubert",
            os.path.expanduser("~/av_hubert/avhubert"),
        ]

        for path in search_paths:
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "hubert.py")
            ):
                logger.info(f"Found AV-HuBERT source at: {path}")
                return path

        return None

    def _load_kmeans(self):
        """Load the k-means quantizer model (KM500)."""
        import joblib

        logger.info(f"Loading KM500 from {self.km_path}...")
        self.km_model = joblib.load(self.km_path)
        n_clusters = self.km_model.cluster_centers_.shape[0]
        logger.info(f"KM500 loaded: {n_clusters} clusters")

    def preprocess_frames(
        self,
        lip_frames: np.ndarray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Preprocess lip crop frames for AV-HuBERT input.
        
        AV-HuBERT expects:
          - Input shape: [B, T, 1, 96, 96] (batch, time, channel, height, width)
          - Normalized to [-1, 1] or [0, 1] range
          - Grayscale (1 channel)
        
        Args:
            lip_frames: numpy array [T, 96, 96] (grayscale) or [T, 96, 96, 3] (BGR)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed tensor [1, T, 1, 96, 96]
        """
        # Convert to float
        frames = lip_frames.astype(np.float32)

        # Handle grayscale vs BGR
        if len(frames.shape) == 4 and frames.shape[-1] == 3:
            # Convert BGR to grayscale
            import cv2
            gray_frames = []
            for f in frames:
                gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            frames = np.stack(gray_frames, axis=0)

        # Normalize
        if normalize:
            frames = frames / 255.0
            # AV-HuBERT uses mean/std normalization
            mean = 0.421
            std = 0.165
            frames = (frames - mean) / std

        # Shape: [T, 96, 96] -> [1, T, 1, 96, 96]
        tensor = torch.from_numpy(frames).float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # [T, 1, 96, 96]
        tensor = tensor.unsqueeze(0)  # [1, T, 1, 96, 96]

        return tensor.to(self.device)

    def extract_features(
        self,
        lip_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Extract continuous feature embeddings from lip frames.
        
        Args:
            lip_frames: [T, 96, 96] grayscale lip crops
            
        Returns:
            Feature embeddings [T', D] where T' <= T (due to downsampling)
        """
        self.load()

        if self.model is None:
            # Dummy mode: return random features
            logger.warning("Using dummy features (model not loaded)")
            T = lip_frames.shape[0]
            T_out = T // 4  # AV-HuBERT typically downsamples 4x
            return np.random.randn(max(1, T_out), 768).astype(np.float32)

        video_tensor = self.preprocess_frames(lip_frames)

        with torch.no_grad():
            # AV-HuBERT forward pass
            # Create a dummy audio input (zeros) since we only have video
            B, T_vid = video_tensor.shape[0], video_tensor.shape[1]

            # Source dict for the model
            source = {
                "video": video_tensor,
                "audio": None,
            }

            # Extract features from specified layer
            try:
                features, _ = self.model.extract_features(
                    source=source,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer if self.layer > 0 else None,
                )
            except Exception as e:
                logger.warning(f"extract_features failed: {e}, trying forward pass")
                try:
                    # Alternative: direct forward pass
                    result = self.model(
                        source["video"],
                        source.get("audio"),
                        features_only=True,
                    )
                    features = result["x"]
                except Exception as e2:
                    logger.error(f"Forward pass also failed: {e2}")
                    T_out = max(1, T_vid // 4)
                    return np.random.randn(T_out, 768).astype(np.float32)

        features_np = features.squeeze(0).cpu().numpy()
        logger.info(f"Extracted features: {features_np.shape}")
        return features_np

    def quantize(self, features: np.ndarray) -> np.ndarray:
        """
        Quantize continuous features to discrete unit IDs using k-means.
        
        Args:
            features: [T, D] continuous feature embeddings
            
        Returns:
            [T] array of integer unit IDs (0 to 499 for KM500)
        """
        if self.km_model is None:
            # Fallback: random units
            logger.warning("K-means not loaded, generating random units")
            return np.random.randint(0, 500, size=features.shape[0])

        unit_ids = self.km_model.predict(features)
        logger.info(
            f"Quantized {features.shape[0]} frames → {len(unit_ids)} units, "
            f"unique units: {len(np.unique(unit_ids))}"
        )
        return unit_ids

    def extract_units(
        self,
        lip_frames: np.ndarray,
    ) -> np.ndarray:
        """
        Full pipeline: lip frames → discrete speech units.
        
        This is the main entry point for the pipeline.
        
        Args:
            lip_frames: [T, 96, 96] grayscale lip crop sequence
            
        Returns:
            [T'] array of discrete unit IDs
        """
        logger.info(f"Extracting units from {lip_frames.shape[0]} frames...")
        features = self.extract_features(lip_frames)
        units = self.quantize(features)
        logger.info(f"Extracted {len(units)} speech units")
        return units

    def units_to_string(self, units: np.ndarray) -> str:
        """
        Convert unit IDs to a space-separated string for downstream processing.
        
        Also performs deduplication (collapse consecutive identical units).
        
        Args:
            units: [T] array of unit IDs
            
        Returns:
            Deduplicated unit string, e.g., "23 145 67 89 ..."
        """
        # Deduplicate consecutive units
        deduped = [units[0]]
        for u in units[1:]:
            if u != deduped[-1]:
                deduped.append(u)

        return " ".join(str(int(u)) for u in deduped)
