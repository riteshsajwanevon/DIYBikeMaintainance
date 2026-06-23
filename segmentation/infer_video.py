import cv2
from ultralytics import YOLO
import numpy as np

def test_on_video_custom(video_path):
    model = YOLO('best.pt')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    print(f"Starting custom video inference on: {video_path}")
    print("Press 'q' in the video window to stop early!\n")

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video
            
        frame_count += 1
        
        # Run YOLO on the single frame
        # verbose=False stops it from printing the speed on every single frame
        # conf=0.50 automatically filters out ANY masks with less than 50% confidence!
        results = model(frame, verbose=False, conf=0.50)
        result = results[0]
        # Draw the masks automatically, but hide the boxes
        annotated_frame = result.plot(boxes=False)
        
        # If masks are found, let's add custom labels and print the pixels!
        if result.masks is not None:
            # Print a header for every 30th frame so the terminal doesn't get too crazy
            if frame_count % 30 == 0:
                print(f"--- Frame {frame_count} ---")

            polygons = result.masks.xy
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, polygon in enumerate(polygons):
                class_id = int(classes[i])
                class_name = model.names[class_id]
                conf = confidences[i]
                
                # Create a label string that includes the confidence (e.g. "engine-oil-window 0.95")
                label_text = f"{class_name} {conf:.2f}"
                
                # Print pixel data to terminal (only every 30 frames to avoid spam)
                if frame_count % 30 == 0:
                    print(f"  Detected: {label_text} | Polygon Size: {len(polygon)} points")
                    if len(polygon) > 0:
                        print(f"  First point [x, y]: {polygon[0]}")
                
                # Draw the label directly on the image!
                if len(polygon) > 0:
                    # Find the center of the polygon using OpenCV moments
                    M = cv2.moments(polygon)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # Draw the main label at the center
                        cv2.putText(annotated_frame, label_text, (cX, cY), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) # Black outline
                        cv2.putText(annotated_frame, label_text, (cX, cY), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # White text
                        
                        # Draw the pixel data right below it!
                        pixel_text = f"Pixels: {len(polygon)} | Start: [{int(polygon[0][0])}, {int(polygon[0][1])}]"
                        cY_offset = cY + 30 # Move down by 30 pixels for the second line
                        
                        cv2.putText(annotated_frame, pixel_text, (cX, cY_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3) # Black outline
                        cv2.putText(annotated_frame, pixel_text, (cX, cY_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text

        # Show the video frame on the screen
        cv2.imshow("YOLOv11 Custom Segmentation", annotated_frame)
        
        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Video inference finished!")

if __name__ == '__main__':
    video_to_test = '../Test3.mp4'  
    test_on_video_custom(video_to_test)
