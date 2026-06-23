import json
import os
import glob
import numpy as np
import cv2
try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("pycocotools is required. Please install it first.")
    exit(1)

def convert_roboflow_coco_to_yolo():
    print("Scanning for Roboflow COCO annotations...")
    json_files = glob.glob('**/_annotations.coco.json', recursive=True)
    
    if not json_files:
        print("Error: No '_annotations.coco.json' files found.")
        return

    for json_path in json_files:
        print(f"\nProcessing: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        cat_id_to_yolo_id = {}
        for i, cat in enumerate(data['categories']):
            cat_id_to_yolo_id[cat['id']] = i
            
        images = {img['id']: img for img in data['images']}
        img_annotations = {img['id']: [] for img in data['images']}
        
        for ann in data['annotations']:
            image_id = ann['image_id']
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_yolo_id:
                continue
                
            yolo_class_id = cat_id_to_yolo_id[cat_id]
            img = images[image_id]
            img_w, img_h = img['width'], img['height']
            
            seg = ann.get('segmentation', [])
            polygons = []
            
            if isinstance(seg, dict):
                # Handle RLE format from Roboflow
                if isinstance(seg['counts'], list):
                    rle = maskUtils.frPyObjects([seg], img_h, img_w)[0]
                else:
                    rle = seg
                mask = maskUtils.decode(rle)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if contour.shape[0] >= 3: # Need at least 3 points for a polygon
                        polygons.append(contour.flatten().tolist())
            elif isinstance(seg, list):
                # Handle standard polygon format
                polygons = seg
                
            for polygon in polygons:
                if len(polygon) < 6: # Ensure we have at least x1,y1,x2,y2,x3,y3
                    continue
                normalized_poly = []
                for i in range(0, len(polygon), 2):
                    x = float(polygon[i]) / img_w
                    y = float(polygon[i+1]) / img_h
                    normalized_poly.append(f"{x:.6f} {y:.6f}")
                
                if normalized_poly:
                    line = f"{yolo_class_id} " + " ".join(normalized_poly)
                    img_annotations[image_id].append(line)
        
        folder_path = os.path.dirname(json_path)
        count = 0
        for img_id, lines in img_annotations.items():
            if lines: # Only write if there are annotations
                base_name = os.path.splitext(images[img_id]['file_name'])[0]
                with open(os.path.join(folder_path, f"{base_name}.txt"), 'w') as f:
                    f.write("\n".join(lines))
                count += 1
                
        print(f"  Success: Generated {count} YOLO .txt files in {folder_path}/")

if __name__ == "__main__":
    convert_roboflow_coco_to_yolo()
