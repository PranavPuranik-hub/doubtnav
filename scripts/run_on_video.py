import sys
import os
import cv2
import time
from pathlib import Path

# Add backend to path so we can import perception
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.perception import Perceiver

def process_video(input_path, output_path):
    print(f"Loading Perceiver model...")
    # Target >= 6 FPS, CPU
    perceiver = Perceiver(device="cpu")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video {input_path}")
        return
        
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (512, 384))
    
    frame_count = 0
    start_time = time.time()
    
    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        group_mask, uncertainty, overlay = perceiver.predict(frame)
        t1 = time.time()
        
        fps = 1.0 / (t1 - t0)
        print(f"Frame {frame_count}: Processed at {fps:.2f} FPS")
        
        out.write(overlay)
        frame_count += 1
        
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nFinished processing {frame_count} frames.")
    print(f"Average FPS: {avg_fps:.2f} (Target was >= 6 FPS)")
    print(f"Saved output to {output_path}")
    
    cap.release()
    out.release()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_on_video.py <input_video> <output_video>")
    else:
        process_video(sys.argv[1], sys.argv[2])
