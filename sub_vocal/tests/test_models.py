"""Unit tests for model architectures."""

import torch
import pytest


def test_visual_encoder_shape():
    from sub_vocal.models.visual_encoder import VisualEncoder
    enc = VisualEncoder(backbone="resnet18", pretrained_2d=False, out_dim=512)
    video = torch.randn(2, 16, 1, 96, 96)
    mask = torch.ones(2, 16, dtype=torch.bool)
    out, out_mask = enc(video, mask)
    assert out.shape[0] == 2
    assert out.shape[2] == 512
    assert out.dim() == 3


def test_mel_decoder_teacher_forced():
    from sub_vocal.models.decoder import MelDecoder
    dec = MelDecoder(d_model=256, nhead=4, num_layers=2, mel_dim=80)
    enc_out = torch.randn(2, 10, 256)
    enc_mask = torch.ones(2, 10, dtype=torch.bool)
    target_mel = torch.randn(2, 20, 80)
    mel_out, mel_post, stop = dec(enc_out, enc_mask, target_mel)
    assert mel_out.shape == (2, 20, 80)
    assert mel_post.shape == (2, 20, 80)
    assert stop.shape == (2, 20, 1)


def test_speaker_encoder():
    from sub_vocal.models.speaker_embed import SpeakerEncoder
    enc = SpeakerEncoder(input_dim=80, hidden_dim=128, embed_dim=64)
    mel = torch.randn(2, 100, 80)
    embed = enc(mel)
    assert embed.shape == (2, 64)
    # Check L2 normalized
    norms = embed.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_film_conditioning():
    from sub_vocal.models.speaker_embed import FiLMLayer
    film = FiLMLayer(feature_dim=256, condition_dim=64)
    features = torch.randn(2, 10, 256)
    condition = torch.randn(2, 64)
    out = film(features, condition)
    assert out.shape == (2, 10, 256)


def test_ctc_head():
    from sub_vocal.models.ctc_head import CTCPhonemeHead
    head = CTCPhonemeHead(input_dim=256, vocab_size=44)
    features = torch.randn(2, 20, 256)
    log_probs = head(features)
    assert log_probs.shape == (2, 20, 44)
    # Check log probabilities sum to ~1
    probs = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_lip2speech_model_phase1():
    from sub_vocal.models.lip2speech_model import Lip2SpeechModel
    model = Lip2SpeechModel(phase=1, backbone="resnet18", pretrained_2d=False, num_classes=10)
    batch = {
        "video": torch.randn(2, 16, 1, 96, 96),
        "video_mask": torch.ones(2, 16, dtype=torch.bool),
        "label": torch.randint(0, 10, (2,)),
    }
    out = model(batch)
    assert "logits" in out
    assert out["logits"].shape == (2, 10)


def test_lip2speech_model_phase2():
    from sub_vocal.models.lip2speech_model import Lip2SpeechModel
    model = Lip2SpeechModel(
        phase=2, backbone="resnet18", pretrained_2d=False,
        encoder_dim=256, decoder_layers=2, decoder_heads=4,
        decoder_ff_dim=512, mel_dim=80, speaker_dim=64,
    )
    batch = {
        "video": torch.randn(2, 16, 1, 96, 96),
        "video_mask": torch.ones(2, 16, dtype=torch.bool),
        "mel": torch.randn(2, 40, 80),
        "mel_mask": torch.ones(2, 40, dtype=torch.bool),
        "text": ["hello world", "test speech"],
    }
    out = model(batch)
    assert "mel_out" in out
    assert "mel_postnet" in out
    assert "ctc_log_probs" in out


def test_hifi_gan_generator():
    from sub_vocal.models.hifi_gan import HiFiGANGenerator
    gen = HiFiGANGenerator(
        upsample_rates=[8, 8, 2, 2],
        upsample_kernel_sizes=[16, 16, 4, 4],
        upsample_initial_channel=128,
    )
    mel = torch.randn(2, 80, 50)
    audio = gen(mel)
    assert audio.shape[0] == 2
    assert audio.shape[1] == 1
    # Output length should be ~50 * 8 * 8 * 2 * 2 = 12800
    assert audio.shape[2] > 0


def test_multitask_loss():
    from sub_vocal.losses.multitask import MultiTaskLoss
    loss_fn = MultiTaskLoss(ctc_weight=0.0)  # Disable CTC for simpler test
    outputs = {
        "mel_out": torch.randn(2, 40, 80),
        "mel_postnet": torch.randn(2, 40, 80),
        "stop_logits": torch.randn(2, 40, 1),
    }
    batch = {
        "mel": torch.randn(2, 40, 80),
        "mel_mask": torch.ones(2, 40, dtype=torch.bool),
    }
    losses = loss_fn(outputs, batch)
    assert "total_loss" in losses
    assert losses["total_loss"].requires_grad


if __name__ == "__main__":
    # Run tests manually
    tests = [
        test_visual_encoder_shape,
        test_mel_decoder_teacher_forced,
        test_speaker_encoder,
        test_film_conditioning,
        test_ctc_head,
        test_lip2speech_model_phase1,
        test_lip2speech_model_phase2,
        test_hifi_gan_generator,
        test_multitask_loss,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
