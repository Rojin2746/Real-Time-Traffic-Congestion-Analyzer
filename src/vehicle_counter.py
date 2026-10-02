from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("data/input/traffic_video.mp4")

# COCO vehicle classes
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

    # Counters for current frame
    car_count = 0
    motorcycle_count = 0
    bus_count = 0
    truck_count = 0

    # Get detected objects
    if results[0].boxes is not None:

        for box in results[0].boxes:

            class_id = int(box.cls[0])

            if class_id == 2:
                car_count += 1

            elif class_id == 3:
                motorcycle_count += 1

            elif class_id == 5:
                bus_count += 1

            elif class_id == 7:
                truck_count += 1

    # Total vehicles in current frame
    total_vehicles = (
        car_count
        + motorcycle_count
        + bus_count
        + truck_count
    )

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Display vehicle counts
    cv2.putText(
        annotated_frame,
        f"Cars: {car_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Motorcycles: {motorcycle_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Buses: {bus_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Trucks: {truck_count}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Total Vehicles: {total_vehicles}",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # Show result
    cv2.imshow("Traffic Vehicle Counter", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()