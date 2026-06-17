from ultralytics import YOLO
import os

def main():
    weights_path = "run1/best.pt"
    
    if not os.path.exists(weights_path):
        print(f"Error: Could not find trained weights at {weights_path}.")
        print("Make sure training has completed successfully.")
        return
        
    print(f"Loading trained model from {weights_path}...")
    model = YOLO(weights_path)
    
    video_path = "Test1.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        return
    
    print(f"Running inference on {video_path}...")
    # Predict and save output
    results = model.predict(
        source=video_path,
        show=True,
        project="runs/segment",
        name="inference_video",
        conf=0.1
    )
    
    print("Inference completed! Check the 'runs/segment/inference_video' directory for the output video.")

if __name__ == "__main__":
    main()
