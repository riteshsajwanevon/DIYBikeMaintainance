from ultralytics import YOLO
import os

def main():
    print("Initializing YOLO11 nano segmentation model...")
    model = YOLO("yolo11n-seg.pt")
    
    yaml_path = os.path.abspath("dataset1/data.yaml")
    
    print(f"Starting training on dataset: {yaml_path}")
    # Train the model
    # Using 30 epochs for a balance between speed and quality
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        project="runs/segment",
        name="train_dataset1"
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()
