from ultralytics import YOLO
import cv2
import torch
import threading
import numpy as np
import sys
import time

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    return writer

original_classNames = ['Hardhat', 'NO-Hardhat']

# Settings for input source
USE_WEBCAM = False
VIDEO_FILE = 'files/videos/cam.mp4'  # Specify the path to the video file

# Use webcam or video file
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)  # Use the correct index for your webcam
else:
    cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

writer = create_video_writer(cap, "live_output.mp4")

# Load model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'best.pt'
model = YOLO(model_path).to(device)

frame_skip = 10  # Process every 10th frame
frame_count = 0
resize_factor = 0.3  # Downscale factor
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_delay = 1 / fps
consecutive_frames_needed = 3  # Number of consecutive frames needed to confirm a class change

# Frame capture in a separate thread
def capture_frames():
    global frame
    while True:
        success, img = cap.read()
        if not success:
            break
        frame = img
        time.sleep(frame_delay)  # Ensure the video plays at normal speed

frame = None
thread = threading.Thread(target=capture_frames)
thread.daemon = True
thread.start()

# For smoothing bounding boxes
prev_boxes = []
last_detected_class = None
consecutive_class_count = 0

class NullWriter:
    def write(self, _): pass
    def flush(self): pass

null_writer = NullWriter()
old_stdout = sys.stdout

while True:
    if frame is None:
        continue

    frame_count += 1
    img = frame.copy()
    
    detected_classes = []  # List to store detected class names

    if frame_count % frame_skip == 0:
        # Define the center third ROI
        height, width, _ = img.shape
        roi_x1, roi_x2 = width // 3, 2 * width // 3
        roi = img[:, roi_x1:roi_x2]

        # Downscale the ROI
        small_roi = cv2.resize(roi, (0, 0), fx=resize_factor, fy=resize_factor)

        sys.stdout = null_writer  # Redirect stdout to null writer
        results = model(small_roi, stream=True, verbose=False)
        sys.stdout = old_stdout  # Restore original stdout

        new_boxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1 / resize_factor), int(y1 / resize_factor), int(x2 / resize_factor), int(y2 / resize_factor)
                x1 += roi_x1  # Adjust x coordinates to the original frame
                x2 += roi_x1
                new_boxes.append((x1, y1, x2, y2, int(box.cls[0]), box.conf[0]))

                if box.conf[0] > 0.5:
                    currentClass = original_classNames[int(box.cls[0])]
                    detected_classes.append(currentClass)
        
        prev_boxes = new_boxes

        # Determine the current detected class
        current_detected_class = None
        if 'NO-Hardhat' in detected_classes:
            current_detected_class = 'NO-Hardhat'
        elif 'Hardhat' in detected_classes:
            current_detected_class = 'Hardhat'

        # Buffer mechanism to confirm class change
        if current_detected_class == last_detected_class:
            consecutive_class_count = 0
        elif current_detected_class is not None:
            if current_detected_class != last_detected_class:
                consecutive_class_count += 1
                if consecutive_class_count >= consecutive_frames_needed:
                    last_detected_class = current_detected_class
                    consecutive_class_count = 0
            else:
                consecutive_class_count = 0
    else:
        # Use previous detections if no new detections are made
        for (x1, y1, x2, y2, cls, conf) in prev_boxes:
            if conf > 0.5:
                currentClass = original_classNames[cls]
                detected_classes.append(currentClass)

        # Update current detected class based on previous detections
        if 'NO-Hardhat' in detected_classes:
            current_detected_class = 'NO-Hardhat'
        elif 'Hardhat' in detected_classes:
            current_detected_class = 'Hardhat'
        else:
            current_detected_class = None

        # Buffer mechanism to confirm class change (same as above)
        if current_detected_class == last_detected_class:
            consecutive_class_count = 0
        elif current_detected_class is not None:
            if current_detected_class != last_detected_class:
                consecutive_class_count += 1
                if consecutive_class_count >= consecutive_frames_needed:
                    last_detected_class = current_detected_class
                    consecutive_class_count = 0
            else:
                consecutive_class_count = 0

    # Display the last confirmed detected class
    if last_detected_class == 'NO-Hardhat':
        cv2.putText(img, 'X', (width - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    elif last_detected_class == 'Hardhat':
        cv2.putText(img, 'O', (width - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
    
    writer.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
