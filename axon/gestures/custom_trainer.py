import os
import argparse
import time
import json
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .custom_model import HandGestureMLP
from ..utils.logger import logger

def normalize_landmarks(landmarks) -> np.ndarray:
    """Normalizes hand landmarks by shifting to wrist origin and scaling by bounding box size."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = pts[0]
    normalized = pts - wrist
    
    # Scale normalization
    min_vals = np.min(normalized, axis=0)
    max_vals = np.max(normalized, axis=0)
    scale = np.max(max_vals - min_vals)
    if scale == 0:
        scale = 1.0
        
    normalized = normalized / scale
    return normalized.flatten()  # shape (63,)


def main():
    parser = argparse.ArgumentParser(description="AXON Custom Gesture Recorder & Trainer")
    parser.add_argument("--name", type=str, required=True, help="Name of the gesture to record/train")
    args = parser.parse_args()
    
    gesture_name = args.name.upper()
    
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    saved_models_dir = os.path.join(current_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    
    dataset_file = os.path.join(saved_models_dir, "gesture_dataset.json")
    model_path = os.path.join(saved_models_dir, f"{gesture_name.lower()}.pt")
    labels_file = os.path.join(saved_models_dir, "labels.yaml")
    
    # Load model task path
    axon_dir = os.path.dirname(current_dir)
    hand_model_task = os.path.join(axon_dir, "models", "hand_landmarker.task")
    
    # 1. RECORDING SAMPLES
    print(f"=== RECORDING CUSTOM GESTURE: {gesture_name} ===")
    print("Hold the gesture steady in front of the camera.")
    print("Press SPACEBAR to start recording 40 samples (takes ~3 seconds).")
    print("Press ESC to exit.")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open laptop camera.")
        return
        
    # Load MediaPipe Hand Landmarker for recording
    base_options = python.BaseOptions(model_asset_path=hand_model_task)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    samples = []
    recording = False
    
    while len(samples) < 40:
        ret, frame = cap.read()
        if not ret:
            continue
            
        h, w, _ = frame.shape
        display_frame = cv2.flip(frame, 1)  # Mirror for natural view
        
        cv2.putText(display_frame, f"Samples recorded: {len(samples)}/40", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        if recording:
            # Process current frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect(mp_img)
            
            if res.hand_landmarks:
                landmarks = res.hand_landmarks[0]
                norm_feat = normalize_landmarks(landmarks)
                samples.append(norm_feat.tolist())
                # Flash circle to show sample capture
                cv2.circle(display_frame, (w - 30, 30), 10, (0, 0, 255), -1)
                time.sleep(0.08)  # Sample interval delay
            else:
                cv2.putText(display_frame, "NO HAND DETECTED", (20, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "READY. PRESS SPACEBAR TO RECORD", (20, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
        cv2.imshow("Custom Gesture Recorder", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = True
        elif key == 27:  # ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    
    if len(samples) < 40:
        print("Recording aborted.")
        return
        
    print(f"Recorded {len(samples)} samples successfully.")
    
    # 2. DATASET PREPARATION & LABEL UPDATES
    dataset = {}
    if os.path.exists(dataset_file):
        try:
            with open(dataset_file, 'r') as f:
                dataset = json.load(f)
        except Exception:
            pass
            
    dataset[gesture_name] = samples
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f)
        
    # Read existing labels or create
    labels = []
    if os.path.exists(labels_file):
        try:
            with open(labels_file, 'r') as f:
                labels_dict = yaml.safe_load(f)
                labels = labels_dict.get("labels", [])
        except Exception:
            pass
            
    if gesture_name not in labels:
        labels.append(gesture_name)
        with open(labels_file, 'w') as f:
            yaml.safe_dump({"labels": labels}, f)
            
    print(f"Saved gesture samples to dataset. Labels: {labels}")
    
    # 3. PYTORCH MLP MODEL TRAINING
    print("=== TRAINING PYTORCH MLP MODEL ===")
    
    # Create classes
    num_classes = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    X_train = []
    y_train = []
    
    for gname, gsamples in dataset.items():
        if gname in label_to_idx:
            idx = label_to_idx[gname]
            for s in gsamples:
                X_train.append(s)
                y_train.append(idx)
                
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    model = HandGestureMLP(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Simple training loop
    model.train()
    for epoch in range(100):
        # Forward pass
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
            
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print("Training Complete!")


if __name__ == "__main__":
    main()
