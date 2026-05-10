"""
Lip2Speech Full Model
=======================

Assembles all components into a single nn.Module:
  - Visual encoder (3D-ResNet)
  - Mel decoder (Transformer)
  - Speaker embedding (GE2E + FiLM)
  - CTC phoneme head

Supports Phase 1 (classification), Phase 2 (full lip2speech), 
and Phase 4 (multilingual adapter) modes.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

from .visual_encoder import VisualEncoder
from .decoder import MelDecoder
from .speaker_embed import SpeakerEmbedding
from .ctc_head import CTCPhonemeHead

logger = logging.getLogger(__name__)


class Lip2SpeechModel(nn.Module):
    """
    Complete lip-to-speech synthesis model.
    
    Phase 1: visual_encoder + classifier → word classification
    Phase 2: visual_encoder + decoder + speaker_embed + ctc_head → mel synthesis
    """

    def __init__(
        self,
        phase: int = 2,
        # Visual encoder
        backbone: str = "resnet18",
        pretrained_2d: bool = True,
        encoder_dim: int = 512,
        lrw_pretrained_path: Optional[str] = None,
        # Decoder
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        mel_dim: int = 80,
        # Speaker
        speaker_dim: int = 256,
        speaker_method: str = "film",
        # CTC
        ctc_vocab_size: int = 44,
        ctc_enabled: bool = True,
        # Classifier (Phase 1)
        num_classes: int = 500,
        # Multilingual (Phase 4)
        num_languages: int = 0,
        language_dim: int = 64,
    ):
        super().__init__()
        self.phase = phase

        # ── Visual Encoder (all phases) ──
        self.visual_encoder = VisualEncoder(
            backbone=backbone,
            pretrained_2d=pretrained_2d,
            out_dim=encoder_dim,
            lrw_pretrained_path=lrw_pretrained_path,
        )

        if phase == 1:
            # ── Phase 1: Classification head ──
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(encoder_dim, num_classes),
            )
        else:
            # ── Phase 2+: Full lip2speech ──
            self.decoder = MelDecoder(
                d_model=encoder_dim,
                nhead=decoder_heads,
                num_layers=decoder_layers,
                dim_feedforward=decoder_ff_dim,
                mel_dim=mel_dim,
                speaker_dim=speaker_dim if speaker_method != "film" else 0,
            )

            self.speaker_embed = SpeakerEmbedding(
                embed_dim=speaker_dim,
                feature_dim=encoder_dim,
                method=speaker_method,
                num_film_layers=decoder_layers,
            )

            if ctc_enabled:
                self.ctc_head = CTCPhonemeHead(
                    input_dim=encoder_dim,
                    vocab_size=ctc_vocab_size,
                )
            else:
                self.ctc_head = None

            # Phase 4: Language embedding
            if num_languages > 0:
                self.language_embed = nn.Embedding(num_languages, language_dim)
                self.language_proj = nn.Linear(language_dim, encoder_dim)
            else:
                self.language_embed = None

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Lip2SpeechModel Phase {phase}: "
            f"{total_params/1e6:.1f}M total, {trainable/1e6:.1f}M trainable"
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the appropriate phase.
        
        Args:
            batch: Dict with keys: video, video_mask, mel, mel_mask, etc.
        Returns:
            Dict of outputs (logits, mel predictions, etc.)
        """
        video = batch["video"]            # [B, T, 1, H, W]
        video_mask = batch.get("video_mask")

        # Visual encoding
        enc_out, enc_mask = self.visual_encoder(video, video_mask)

        if self.phase == 1:
            return self._forward_phase1(enc_out, enc_mask, batch)
        else:
            return self._forward_phase2(enc_out, enc_mask, batch)

    def _forward_phase1(self, enc_out, enc_mask, batch):
        """Phase 1: classification."""
        # Pool over time → classify
        x = enc_out.permute(0, 2, 1)  # [B, D, T]
        logits = self.classifier(x)    # [B, num_classes]
        return {"logits": logits}

    def _forward_phase2(self, enc_out, enc_mask, batch):
        """Phase 2: full lip-to-speech."""
        outputs = {}

        # Speaker embedding from ground-truth audio mel
        speaker_emb = None
        if "mel" in batch and batch["mel"].dim() == 3:
            speaker_emb = self.speaker_embed.encode(batch["mel"])
            # Apply FiLM conditioning to encoder output
            if self.speaker_embed.method == "film":
                enc_out = self.speaker_embed.condition(enc_out, speaker_emb, layer_idx=0)

        # Language conditioning
        if self.language_embed is not None and "language_id" in batch:
            lang_emb = self.language_embed(batch["language_id"])
            enc_out = enc_out + self.language_proj(lang_emb).unsqueeze(1)

        # Mel decoder (teacher-forced)
        target_mel = batch.get("mel")
        target_mask = batch.get("mel_mask")
        if target_mel is not None and target_mel.dim() == 3:
            mel_out, mel_postnet, stop_logits = self.decoder(
                enc_out, enc_mask, target_mel, target_mask, speaker_emb
            )
            outputs["mel_out"] = mel_out
            outputs["mel_postnet"] = mel_postnet
            outputs["stop_logits"] = stop_logits

        # CTC head
        if self.ctc_head is not None:
            ctc_log_probs = self.ctc_head(enc_out)
            outputs["ctc_log_probs"] = ctc_log_probs

        outputs["encoder_out"] = enc_out
        outputs["speaker_embed"] = speaker_emb

        return outputs

    def inference(
        self,
        video: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        speaker_mel: Optional[torch.Tensor] = None,
        max_mel_len: int = 1500,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode (no teacher forcing).
        
        Args:
            video: [B, T, 1, H, W]
            video_mask: [B, T]
            speaker_mel: [B, T_spk, mel_dim] reference mel for speaker
            max_mel_len: Maximum mel frames to generate
        Returns:
            Dict with 'mel_postnet' and 'stop_probs'
        """
        self.eval()
        with torch.no_grad():
            enc_out, enc_mask = self.visual_encoder(video, video_mask)

            speaker_emb = None
            if speaker_mel is not None:
                speaker_emb = self.speaker_embed.encode(speaker_mel)
                if self.speaker_embed.method == "film":
                    enc_out = self.speaker_embed.condition(enc_out, speaker_emb)

            mel_postnet, stop_probs = self.decoder.inference(
                enc_out, enc_mask, speaker_emb, max_len=max_mel_len
            )

        return {"mel_postnet": mel_postnet, "stop_probs": stop_probs}

    def freeze_encoder(self):
        """Freeze the visual encoder (for Phase 2 after Phase 1 pretraining)."""
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        logger.info("Visual encoder frozen")

    def load_encoder_weights(self, checkpoint_path: str):
        """Load visual encoder weights from a Phase 1 checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        encoder_state = {}
        for k, v in state_dict.items():
            if k.startswith("visual_encoder."):
                encoder_state[k.replace("visual_encoder.", "")] = v

        missing, unexpected = self.visual_encoder.load_state_dict(encoder_state, strict=False)
        logger.info(
            f"Loaded encoder from {checkpoint_path}: "
            f"{len(encoder_state)} keys, {len(missing)} missing, {len(unexpected)} unexpected"
        )
