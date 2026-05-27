import os
import urllib.request

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {os.path.basename(dest_path)}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axon", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs
    models = {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    }
    
    for filename, url in models.items():
        dest = os.path.join(models_dir, filename)
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print(f"{filename} already exists at {dest}")

if __name__ == "__main__":
    main()
