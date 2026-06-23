from ultralytics import YOLO
import cv2
import numpy as np

def run_inference(image_path):
    # 1. Load your newly trained model
    model = YOLO('best.pt')

    # 2. Run prediction
    # Setting boxes=False hides the bounding boxes from the saved image!
    results = model.predict(source=image_path, save=True, boxes=False)

    # 3. Extracting the actual Pixels!
    # The 'results' object contains the raw data. We get the first result (since we only passed one image).
    result = results[0]

    # Check if the model actually found any masks in the image
    if result.masks is not None:
        print("\n--- Mask Pixel Data Extracted! ---")
        
        # result.masks.xy contains the polygon coordinates for each detected object
        polygons = result.masks.xy
        
        # result.boxes.cls contains the class ID (0, 1, 2...) for each detected object
        classes = result.boxes.cls.cpu().numpy()
        
        # Get the class names dictionary (e.g. {0: 'DIY-Bike-Maintenance', 2: 'engine-casing-left', ...})
        class_names = model.names

        for i, polygon in enumerate(polygons):
            class_id = int(classes[i])
            class_name = class_names[class_id]
            
            # The polygon is a list of [x, y] pixel coordinates outlining the object!
            num_points = len(polygon)
            print(f"Detected: {class_name}")
            print(f" -> Made of a polygon with {num_points} pixel points.")
            
            # Example: Print the first 3 pixel coordinates [x, y] of this mask
            print(f" -> First 3 points (x, y): {polygon[:3]}\n")
            
            # If you want to isolate the pixels (e.g., to check the oil color inside the window),
            # you can use these polygon points with OpenCV to crop out just that specific shape!
            
    else:
        print("No masks were detected in this image.")

if __name__ == '__main__':
    import glob
    # Automatically find the first image in your test folder!
    test_images = glob.glob('../segmentation_split/test/images/*.jpg')
    
    if test_images:
        test_image_path = test_images[0]
        print(f"Automatically selected test image: {test_image_path}")
        run_inference(test_image_path)
    else:
        print("Could not find any images in the test folder!")
