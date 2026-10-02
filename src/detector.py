from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("data/input/traffic_video.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Display video
    cv2.imshow("Traffic Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()