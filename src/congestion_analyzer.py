import cv2
import os
import math

from ultralytics import YOLO


# ============================================================
# YOLO MODEL
# ============================================================

MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)


# ============================================================
# VEHICLE CLASSES
# COCO CLASS IDs
# ============================================================

VEHICLE_CLASSES = {
    2: "cars",
    3: "motorcycles",
    5: "buses",
    7: "trucks"
}


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""

    return math.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
    )


# ============================================================
# TRAFFIC ALERT
# ============================================================

def get_traffic_alert(congestion):
    """Return an alert based on congestion level."""

    alerts = {
        "LOW": {
            "title": "🟢 NORMAL TRAFFIC",
            "message": "Traffic is flowing normally.",
            "type": "success"
        },

        "MEDIUM": {
            "title": "🟡 MODERATE TRAFFIC",
            "message": "Traffic is moderately congested.",
            "type": "warning"
        },

        "HIGH": {
            "title": "🟠 HIGH TRAFFIC",
            "message": "High traffic congestion detected.",
            "type": "warning"
        },

        "SEVERE": {
            "title": "🔴 SEVERE TRAFFIC",
            "message": "Severe traffic congestion detected.",
            "type": "error"
        }
    }

    return alerts.get(
        congestion,
        alerts["LOW"]
    )


# ============================================================
# ANALYZE TRAFFIC VIDEO
# ============================================================

def analyze_video(
    video_path,
    output_path,
    max_frames=300,
    progress_callback=None,
    data_callback=None
):

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open the traffic video."
        )


    # --------------------------------------------------------
    # VIDEO INFORMATION
    # --------------------------------------------------------

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30
    # ==========================================
    # ACTUAL VIDEO FRAME COUNT
    # ==========================================

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_video_frames <= 0:
        total_video_frames = max_frames


    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    # --------------------------------------------------------
    # OUTPUT VIDEO
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )


    # --------------------------------------------------------
    # INITIAL VARIABLES
    # --------------------------------------------------------

    frame_count = 0

    vehicle_history = []

    previous_positions = {}

    speed_history = []


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # ======================================
        # UPDATE PROCESSING PROGRESS
        # ======================================

        if progress_callback is not None:

            progress = min(
                frame_count / total_video_frames,
                1.0
            )

            progress_callback(
                progress,
                frame_count,
                total_video_frames
            )

        
        # ----------------------------------------------------
        # FRAME LIMIT
        # ----------------------------------------------------

        frame_count += 1

        if max_frames is not None and frame_count >= max_frames:
            break


        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )


        # ----------------------------------------------------
        # VEHICLE COUNTERS
        # ----------------------------------------------------

        cars = 0
        motorcycles = 0
        buses = 0
        trucks = 0

        current_positions = {}


        # ====================================================
        # PROCESS YOLO DETECTIONS
        # ====================================================

        for result in results:

            boxes = result.boxes

            if boxes is None:
                continue


            for box in boxes:

                class_id = int(
                    box.cls[0]
                )


                # Ignore non-vehicle objects

                if class_id not in VEHICLE_CLASSES:
                    continue


                vehicle_type = VEHICLE_CLASSES[
                    class_id
                ]


                # ------------------------------------------------
                # COUNT VEHICLE
                # ------------------------------------------------

                if vehicle_type == "cars":
                    cars += 1

                elif vehicle_type == "motorcycles":
                    motorcycles += 1

                elif vehicle_type == "buses":
                    buses += 1

                elif vehicle_type == "trucks":
                    trucks += 1


                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )


                # ------------------------------------------------
                # CENTER POINT
                # ------------------------------------------------

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # ------------------------------------------------
                # TRACK ID
                # ------------------------------------------------

                track_id = None

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                    current_positions[
                        track_id
                    ] = (
                        center_x,
                        center_y
                    )


                # ------------------------------------------------
                # DRAW BOUNDING BOX
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )


                # ------------------------------------------------
                # DRAW LABEL
                # ------------------------------------------------

                label = vehicle_type

                if track_id is not None:
                    label += (
                        f" ID:{track_id}"
                    )


                cv2.putText(
                    frame,
                    label,
                    (
                        int(x1),
                        max(int(y1) - 10, 20)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # TOTAL VEHICLES
        # ====================================================

        total = (
            cars
            + motorcycles
            + buses
            + trucks
        )


        # ====================================================
        # SPEED ESTIMATION
        # ====================================================

        frame_speeds = []


        for track_id, current_point in (
            current_positions.items()
        ):

            if track_id not in previous_positions:
                continue


            previous_point = previous_positions[
                track_id
            ]


            distance_pixels = calculate_distance(
                previous_point,
                current_point
            )


            # Estimated speed
            speed = (
                distance_pixels
                * fps
                * 0.05
            )


            # Limit unrealistic values
            speed = min(
                speed,
                100
            )


            frame_speeds.append(
                speed
            )


        # ----------------------------------------------------
        # UPDATE TRACKING POSITIONS
        # ----------------------------------------------------

        previous_positions = (
            current_positions.copy()
        )


        # ====================================================
        # AVERAGE SPEED
        # ====================================================

        if frame_speeds:

            average_speed = (
                sum(frame_speeds)
                / len(frame_speeds)
            )

            speed_history.append(
                average_speed
            )

        elif speed_history:

            average_speed = (
                sum(speed_history)
                / len(speed_history)
            )

        else:

            average_speed = 0


        # ====================================================
        # CONGESTION SCORE
        # ====================================================

        density_score = min(
            total / 30,
            1.0
        )


        speed_score = max(
            0,
            min(
                (30 - average_speed) / 30,
                1.0
            )
        )


        congestion_score = (
            0.6 * density_score
            + 0.4 * speed_score
        )


        # ====================================================
        # CONGESTION LEVEL
        # ====================================================

        if congestion_score < 0.25:

            congestion = "LOW"

        elif congestion_score < 0.50:

            congestion = "MEDIUM"

        elif congestion_score < 0.75:

            congestion = "HIGH"

        else:

            congestion = "SEVERE"

        # ==========================================
        # SEND LIVE TRAFFIC DATA
        # ==========================================

        if data_callback is not None:

            live_data = {
                "frame": frame_count,
                "cars": cars,
                "motorcycles": motorcycles,
                "buses": buses,
                "trucks": trucks,
                "total": total,
                "average_speed": average_speed,
                "congestion_score": congestion_score,
                "congestion": congestion
            }

            data_callback(live_data)

        # ====================================================
        # SAVE VEHICLE HISTORY
        # ====================================================

        vehicle_history.append({

            "frame": frame_count,

            "cars": cars,

            "motorcycles": motorcycles,

            "buses": buses,

            "trucks": trucks,

            "total": total,

            "congestion": congestion,

            "congestion_score": congestion_score,

            "average_speed": average_speed

        })


        # ====================================================
        # DISPLAY INFORMATION ON VIDEO
        # ====================================================

        cv2.putText(
            frame,
            f"Vehicles: {total}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Speed: {average_speed:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Congestion: {congestion}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


        # ====================================================
        # WRITE FRAME
        # ====================================================

        out.write(frame)


    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()
    out.release()


    # ========================================================
    # FINAL AVERAGE SPEED
    # ========================================================

    if speed_history:

        final_average_speed = (
            sum(speed_history)
            / len(speed_history)
        )

    else:

        final_average_speed = 0


    # ========================================================
    # FINAL VEHICLE DATA
    # ========================================================

    if vehicle_history:

        last_frame = vehicle_history[-1]

        final_cars = last_frame["cars"]

        final_motorcycles = (
            last_frame["motorcycles"]
        )

        final_buses = (
            last_frame["buses"]
        )

        final_trucks = (
            last_frame["trucks"]
        )

        final_total = last_frame["total"]

        final_congestion = (
            last_frame["congestion"]
        )

    else:

        final_cars = 0
        final_motorcycles = 0
        final_buses = 0
        final_trucks = 0
        final_total = 0
        final_congestion = "LOW"


    # ========================================================
    # CREATE TRAFFIC ALERT
    # ========================================================

    alert = get_traffic_alert(
        final_congestion
    )

    final_congestion_score = 0

    if vehicle_history:
        final_congestion_score = vehicle_history[-1][
            "congestion_score"
        ]


    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "cars": final_cars,

        "motorcycles": final_motorcycles,

        "buses": final_buses,

        "trucks": final_trucks,

        "total": final_total,

        "average_speed": final_average_speed,

        "congestion_score": final_congestion_score,

        "congestion": final_congestion,

        "vehicle_history": vehicle_history,

        "alert": alert

    }