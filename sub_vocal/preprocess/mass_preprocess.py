"""
Mass Dataset Preprocessor
===========================

Orchestrates the extraction of 96x96 canonical lip crops and 16kHz audio 
from raw video datasets (LRW, LRS2, LRS3, MuAViC) using multiprocessing.

Usage:
    python -m sub_vocal.preprocess.mass_preprocess --dataset lrs3 --input_dir /raw/LRS3 --output_dir /data/LRS3-TED --workers 8
"""

import os
import glob
import logging
import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import cv2
import torchaudio

from .detect import LipDetector
from .align import LipAligner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_single_video(args) -> bool:
    """Worker function for processing a single video."""
    input_path, output_video_path, output_audio_path = args

    # Skip if output already exists (resume capability)
    if os.path.exists(output_video_path) and (not output_audio_path or os.path.exists(output_audio_path)):
        return True

    try:
        # Initialize detector per-process (MediaPipe is not thread-safe)
        detector = LipDetector()
        
        try:
            crops, detections = detector.process_video(input_path, target_size=(96,96))
            aligner = LipAligner(target_size=(96,96))
            aligned_crops = aligner.align_video(crops, detections)
        except Exception as e:
            logger.warning(f"Detection failed for {input_path}: {e}")
            return False
        
        if len(aligned_crops) == 0:
            logger.warning(f"No faces detected in {input_path}")
            return False

        # 2. Save Aligned Video
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 25.0, (96, 96), False)
        for frame in aligned_crops:
            out.write(frame)
        out.release()

        # 3. Extract Audio (if requested)
        if output_audio_path:
            os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
            # Use ffmpeg to reliably extract and resample audio to 16kHz mono
            cmd = (
                f"ffmpeg -y -i \"{input_path}\" -vn -acodec pcm_s16le "
                f"-ar 16000 -ac 1 \"{output_audio_path}\" -loglevel error"
            )
            os.system(cmd)
            
            # Check if audio was actually extracted (some videos might be silent)
            if not os.path.exists(output_audio_path):
                # Create silent audio fallback
                duration = len(aligned_crops) / 25.0
                import torch
                silent = torch.zeros(1, int(duration * 16000))
                torchaudio.save(output_audio_path, silent, 16000)

        # Also copy the text transcript if it exists in the same folder
        text_src = str(input_path).replace(".mp4", ".txt")
        if os.path.exists(text_src):
            text_dst = str(output_video_path).replace(".mp4", ".txt")
            import shutil
            shutil.copy2(text_src, text_dst)

        return True

    except Exception as e:
        logger.error(f"Failed {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser("Mass Data Preprocessor")
    parser.add_argument("--dataset", required=True, choices=["lrw", "lrs3", "muavic"])
    parser.add_argument("--input_dir", required=True, help="Raw dataset path")
    parser.add_argument("--output_dir", required=True, help="Processed dataset destination")
    parser.add_argument("--workers", type=int, default=cpu_count() // 2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning {input_dir} for {args.dataset} dataset...")
    
    tasks = []
    
    if args.dataset == "lrw":
        # LRW: <word>/<split>/<id>.mp4
        for word_dir in input_dir.iterdir():
            if not word_dir.is_dir(): continue
            for split in ["train", "val", "test"]:
                split_dir = word_dir / split
                if not split_dir.exists(): continue
                for vid_path in split_dir.glob("*.mp4"):
                    out_vid = output_dir / word_dir.name / split / vid_path.name
                    tasks.append((str(vid_path), str(out_vid), None)) # LRW doesn't need audio for phase 1
                    
    elif args.dataset == "lrs3":
        # LRS3: <split>/<speaker>/<id>.mp4
        for split in ["pretrain", "trainval", "test"]:
            split_dir = input_dir / split
            if not split_dir.exists(): continue
            for speaker_dir in split_dir.iterdir():
                if not speaker_dir.is_dir(): continue
                for vid_path in speaker_dir.glob("*.mp4"):
                    out_vid = output_dir / split / speaker_dir.name / vid_path.name
                    out_aud = output_dir / split / speaker_dir.name / vid_path.with_suffix(".wav").name
                    tasks.append((str(vid_path), str(out_vid), str(out_aud)))

    elif args.dataset == "muavic":
        # MuAViC: <lang>/<split>/.../*.mp4
        for vid_path in input_dir.rglob("*.mp4"):
            rel_path = vid_path.relative_to(input_dir)
            out_vid = output_dir / rel_path
            out_aud = out_vid.with_suffix(".wav")
            tasks.append((str(vid_path), str(out_vid), str(out_aud)))

    logger.info(f"Found {len(tasks)} videos to process.")
    
    if not tasks:
        return

    # Multiprocessing Pool
    logger.info(f"Starting processing with {args.workers} workers...")
    success_count = 0
    with Pool(args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_video, tasks), total=len(tasks)):
            if result:
                success_count += 1

    logger.info(f"Completed! Successfully processed {success_count}/{len(tasks)} videos.")


if __name__ == "__main__":
    main()
