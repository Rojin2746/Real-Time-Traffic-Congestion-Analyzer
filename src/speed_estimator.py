from ultralytics import YOLO
import cv2
import math
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("data/input/traffic_video.mp4")

# Vehicle classes
vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Store previous positions of vehicles
previous_positions = {}

# Store calculated speeds
vehicle_speeds = {}

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished.")
        break

    # Get video FPS
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        verbose=False
    )

    annotated_frame = results[0].plot()

    if results[0].boxes is not None and results[0].boxes.id is not None:

        boxes = results[0].boxes

        track_ids = boxes.id.int().cpu().tolist()
        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(
            track_ids,
            coordinates,
            class_ids
        ):

            x1, y1, x2, y2 = box

            # Find center of vehicle
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_position = (center_x, center_y)

            # Calculate movement
            if track_id in previous_positions:

                previous_position = previous_positions[track_id]

                distance_pixels = math.sqrt(
                    (center_x - previous_position[0]) ** 2
                    + (center_y - previous_position[1]) ** 2
                )

                # Convert pixel movement to approximate speed
                speed = distance_pixels * fps * 0.05

                vehicle_speeds[track_id] = speed

            # Save current position
            previous_positions[track_id] = current_position

            # Display speed
            speed = vehicle_speeds.get(track_id, 0)

            vehicle_name = vehicle_classes.get(
                class_id,
                "vehicle"
            )

            cv2.putText(
                annotated_frame,
                f"{vehicle_name} ID:{track_id} Speed:{speed:.1f}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

    # Calculate average speed
    if vehicle_speeds:

        average_speed = sum(
            vehicle_speeds.values()
        ) / len(vehicle_speeds)

    else:
        average_speed = 0

    cv2.putText(
        annotated_frame,
        f"Average Speed: {average_speed:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    # Display
    cv2.imshow(
        "Traffic Speed Estimation",
        annotated_frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()