"""
Demo CLI — Sub Vocal Speech Synthesizer
=========================================

Command-line interface for lip-to-speech synthesis.

Usage:
    python -m sub_vocal.demo --video input.mp4 --lang fr
    python -m sub_vocal.demo --video input.mp4 --lang hi --output output_hi.wav
    python -m sub_vocal.demo --video input.mp4 --lang en --speaker-ref speaker.wav
"""

import argparse
import logging
import sys
import os
import pathlib

# Auto-set mediapipe model path relative to project root
project_root = pathlib.Path(__file__).parent.parent
model_path = project_root / 'checkpoints' / 'mediapipe' / 'face_landmarker.task'
if model_path.exists() and 'MEDIAPIPE_MODEL_PATH' not in os.environ:
    os.environ['MEDIAPIPE_MODEL_PATH'] = str(model_path)


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    import sys, io
    # Use UTF-8 wrapper to safely log non-ASCII (Hindi, French accents, etc.)
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(utf8_stdout),
            logging.FileHandler("sub_vocal.log", mode="a", encoding="utf-8"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sub Vocal Speech Synthesizer — Lip-to-Speech Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sub_vocal.demo --video input.mp4 --lang en
  python -m sub_vocal.demo --video input.mp4 --lang fr --output output_fr.wav
  python -m sub_vocal.demo --video input.mp4 --lang hi --tts-backend mms
  python -m sub_vocal.demo --list-languages
        """,
    )

    parser.add_argument("--video", "-v", type=str, help="Path to input video file")
    parser.add_argument("--lang", "-l", type=str, default="en", help="Target language (default: en)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output audio path (default: output_{lang}.wav)")
    parser.add_argument("--speaker-ref", type=str, default=None, help="Speaker reference audio for voice cloning")
    parser.add_argument("--tts-backend", type=str, default="auto", choices=["auto", "xtts", "mms"], help="TTS backend")
    parser.add_argument("--avhubert-ckpt", type=str, default=None, help="AV-HuBERT checkpoint path")
    parser.add_argument("--km-path", type=str, default=None, help="KM500 model path")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--fps", type=float, default=25.0, help="Target FPS for frame extraction")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory for intermediates")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--list-languages", action="store_true", help="List supported languages and exit")

    return parser.parse_args()


def list_languages():
    from sub_vocal.models.tts_engine import XTTS_LANGUAGES, MMS_LANGUAGE_MAP
    print("\n╔══════════════════════════════════════════╗")
    print("║   Sub Vocal — Supported Languages        ║")
    print("╠══════════════════════════════════════════╣")
    print("║                                          ║")
    print("║  XTTS v2 (high quality, voice cloning):  ║")
    for lang in sorted(XTTS_LANGUAGES):
        print(f"║    • {lang:<36} ║")
    print("║                                          ║")
    print("║  MMS-TTS (1100+ langs, no cloning):      ║")
    for code, iso in sorted(MMS_LANGUAGE_MAP.items()):
        print(f"║    • {code:<8} → {iso:<24} ║")
    print("║                                          ║")
    print("╚══════════════════════════════════════════╝")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    if args.list_languages:
        list_languages()
        return

    if not args.video:
        print("Error: --video is required. Use --help for usage.")
        sys.exit(1)

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    output_path = args.output or f"output_{args.lang}.wav"

    # Import pipeline
    from sub_vocal.pipeline import LipToSpeechPipeline, PipelineConfig

    config = PipelineConfig(
        language=args.lang,
        output_path=output_path,
        speaker_wav=args.speaker_ref,
        tts_backend=args.tts_backend,
        avhubert_ckpt=args.avhubert_ckpt,
        km_path=args.km_path,
        max_frames=args.max_frames,
        fps=args.fps,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    print("\n" + "="*60)
    print("  Sub Vocal Speech Synthesizer")
    print("="*60)
    print(f"  Video:    {args.video}")
    print(f"  Language: {args.lang}")
    print(f"  Output:   {output_path}")
    print(f"  Backend:  {args.tts_backend}")
    print("="*60 + "\n")

    pipeline = LipToSpeechPipeline(config)
    result = pipeline.run(args.video)

    # Print results
    print("\n" + "="*60)
    if result.success:
        print("  SUCCESS!")
        print(f"  Output:     {result.output_path}")
        print(f"  Frames:     {result.num_frames}")
        safe_text = (result.decoded_text or '')[:80].encode('cp1252', errors='replace').decode('cp1252')
        print(f"  Text:       {safe_text}")
        print(f"  Total time: {result.duration_seconds:.1f}s")
        print("\n  Timing breakdown:")
        for stage, t in result.timings.items():
            print(f"    {stage:<20} {t:.2f}s")
    else:
        print("  FAILED")
        for err in result.errors:
            print(f"  Error: {err}")
    print("="*60 + "\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
