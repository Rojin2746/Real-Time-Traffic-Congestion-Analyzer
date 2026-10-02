from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("data/input/traffic_video.mp4")

# Vehicle classes from COCO dataset
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished.")
        break

    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        verbose=False
    )

    # Draw tracking information
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Vehicle Tracking", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()