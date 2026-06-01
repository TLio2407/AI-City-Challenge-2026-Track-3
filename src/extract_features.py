import os
import json
import cv2
import math
from ultralytics import YOLO

# OPTIMIZATION 1: Use the Medium model instead of X. 
# It is 3x faster and highly accurate for standard traffic objects.
yolo_model = YOLO('yolo11m.pt') 

def calculate_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)

def cache_video_features(video_dir, output_cache_path):
    cache = {}
    
    for root, _, files in os.walk(video_dir):
        for file in files:
            if not file.endswith('.mp4'): continue
            
            video_path = os.path.join(root, file)
            rel_path = os.path.relpath(video_path, video_dir)
            
            # Get accurate FPS to calculate the stride dynamically
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release() 
            
            # We want to process exactly 2 frames per second
            stride = max(1, int(fps / 2))
            print(f"Extracting: {rel_path} (Stride: {stride})")
            
            # OPTIMIZATION 2 & 3: stream=True, vid_stride, and half=True
            results_generator = yolo_model.track(
                video_path, 
                persist=True, 
                tracker="bytetrack.yaml", 
                verbose=False, 
                stream=True,
                vid_stride=stride, # Completely bypasses decoding intermediate frames
                half=True          # Uses FP16 for 2x faster GPU inference
            )
            
            events = []
            track_history = {} 
            
            # We removed the manual frame_idx check because vid_stride handles it natively.
            for frame_idx, r in enumerate(results_generator):
                # Calculate the actual timestamp in the video based on the stride
                timestamp = (frame_idx * stride) / fps
                
                if r.boxes is not None and r.boxes.id is not None:
                    current_boxes = []
                    
                    for box, track_id, cls in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls):
                        class_name = yolo_model.names[int(cls)]
                        if class_name not in ['car', 'truck', 'bus', 'person', 'motorcycle']: continue
                        
                        track_id = int(track_id)
                        box_coords = [float(b) for b in box]
                        center_x = (box_coords[0] + box_coords[2]) / 2
                        center_y = (box_coords[1] + box_coords[3]) / 2
                        
                        speed = "moving"
                        if track_id in track_history:
                            prev_x, prev_y = track_history[track_id]
                            dist = math.hypot(center_x - prev_x, center_y - prev_y)
                            if dist < 5.0: speed = "stopped"
                            elif dist > 50.0: speed = "moving fast"
                            
                        track_history[track_id] = (center_x, center_y)
                        current_boxes.append({"id": track_id, "cls": class_name, "box": box_coords, "speed": speed})
                        
                        events.append(f"{timestamp:.1f}s: {class_name} ({track_id}) is {speed}.")
                    
                    for i in range(len(current_boxes)):
                        for j in range(i + 1, len(current_boxes)):
                            iou = calculate_iou(current_boxes[i]["box"], current_boxes[j]["box"])
                            if iou > 0.3 and current_boxes[i]["speed"] == "stopped":
                                events.append(f"CRITICAL at {timestamp:.1f}s: Potential collision between {current_boxes[i]['cls']} ({current_boxes[i]['id']}) and {current_boxes[j]['cls']} ({current_boxes[j]['id']}).")
            
            cache[rel_path] = " ".join(events) if events else "No relevant traffic objects detected."
            
            with open(output_cache_path, 'w') as f:
                json.dump(cache, f)

# Execute
cache_video_features("./videos", "yolo_tracking_cache.json")