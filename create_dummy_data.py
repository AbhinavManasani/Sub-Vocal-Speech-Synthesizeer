import os
import cv2
import numpy as np
import torchaudio
import torch

def create_dummy_lrw():
    base_dir = "data/LRW"
    words = ["ABOUT", "BLACK"]
    splits = ["train", "val"]
    
    for word in words:
        for split in splits:
            dir_path = os.path.join(base_dir, word, split)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create 2 dummy videos per split
            for i in range(2):
                vid_path = os.path.join(dir_path, f"{word}_{i:05d}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vid_path, fourcc, 25.0, (96, 96), False)
                # 29 frames of random noise
                for _ in range(29):
                    frame = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
                    out.write(frame)
                out.release()
                print(f"Created {vid_path}")

def create_dummy_lrs3():
    base_dir = "data/LRS3-TED"
    splits = ["trainval", "val"]
    
    for split in splits:
        dir_path = os.path.join(base_dir, split, "speaker1")
        os.makedirs(dir_path, exist_ok=True)
        
        for i in range(2):
            # Video
            vid_path = os.path.join(dir_path, f"vid_{i:04d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_path, fourcc, 25.0, (96, 96), False)
            for _ in range(50):  # 2 seconds
                frame = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
                out.write(frame)
            out.release()
            
            # Audio (16kHz, 2 seconds)
            aud_path = os.path.join(dir_path, f"vid_{i:04d}.wav")
            import soundfile as sf
            audio = np.random.randn(32000)
            sf.write(aud_path, audio, 16000)
            
            # Text
            txt_path = os.path.join(dir_path, f"vid_{i:04d}.txt")
            with open(txt_path, "w") as f:
                f.write("HELLO WORLD THIS IS A TEST")
                
            print(f"Created {vid_path}, {aud_path}, {txt_path}")

if __name__ == "__main__":
    create_dummy_lrw()
    create_dummy_lrs3()
