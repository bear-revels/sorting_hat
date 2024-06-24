from ultralytics import YOLO
import cv2
import math

def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

# Original class names from the model
original_classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                       'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
                       'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

# Filtered class names
filtered_classNames = ['Hardhat', 'NO-Hardhat']

# Create a mapping from original class indices to filtered class indices
class_mapping = {original_classNames.index('Hardhat'): filtered_classNames.index('Hardhat'),
                 original_classNames.index('NO-Hardhat'): filtered_classNames.index('NO-Hardhat')}

cap = cv2.VideoCapture('files/videos/cam.mp4')
writer = create_video_writer(cap, "test.mp4")
model = YOLO("files/models/yolov8n-custom.pt")

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if cls in class_mapping:
                currentClass = filtered_classNames[class_mapping[cls]]
                print(currentClass)
                if conf > 0.5:
                    if currentClass == 'NO-Hardhat':
                        myColor = (0, 0, 255)  # Red color for no hardhat
                    elif currentClass == 'Hardhat':
                        myColor = (0, 255, 0)  # Green color for hardhat

                    img = cv2.putText(img, f'{currentClass}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    writer.write(img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()