# REAL-TIME TRAFFIC CONGESTION ANALYZER

## 1. Introduction

Traffic congestion is a major challenge in modern transportation systems. Traditional traffic monitoring methods often depend on manual observation or dedicated infrastructure. These approaches can be expensive and difficult to scale.

The Real-Time Traffic Congestion Analyzer provides an AI-based approach for analyzing traffic videos. The system uses computer vision techniques to detect and track vehicles and estimate traffic conditions.

The application processes traffic video using YOLO and OpenCV and presents the analysis through an interactive Streamlit dashboard.

---

## 2. Problem Statement

Traffic conditions can change rapidly depending on the number of vehicles and their movement speed. A system that considers only vehicle count may not accurately represent the actual traffic condition.

For example, a road may contain many vehicles while traffic is still moving normally. Conversely, a smaller number of vehicles moving very slowly can indicate congestion.

Therefore, the proposed system considers both **vehicle density and estimated speed** when calculating the congestion score.

---

## 3. Proposed System

The proposed system analyzes uploaded traffic videos using YOLO-based vehicle detection and tracking.

The system:

1. Reads traffic video frames.
2. Detects vehicles.
3. Classifies vehicles.
4. Tracks vehicles between frames.
5. Estimates vehicle movement and speed.
6. Calculates vehicle density.
7. Calculates a congestion score.
8. Determines the congestion level.
9. Generates a traffic alert.
10. Displays the results through a Streamlit dashboard.

---

## 4. Objectives

The objectives are:

* Develop an AI-based traffic monitoring system.
* Detect vehicles automatically.
* Classify cars, motorcycles, buses, and trucks.
* Track vehicles across video frames.
* Estimate vehicle speed.
* Calculate vehicle density.
* Combine density and speed for congestion analysis.
* Generate traffic alerts.
* Provide visual traffic analytics.
* Maintain analysis history.
* Export traffic results as CSV.

---

## 5. System Requirements

### Hardware

Recommended:

* Modern multi-core processor
* Minimum 8 GB RAM
* GPU recommended for faster YOLO processing
* Sufficient storage for videos and processed outputs

### Software

* Python 3.x
* Streamlit
* OpenCV
* Ultralytics YOLO
* Pandas

---

## 6. System Modules

### 6.1 Video Input Module

Allows the user to upload traffic videos in supported formats.

Supported formats include:

```text
MP4
AVI
MOV
```

---

### 6.2 Vehicle Detection Module

YOLO is used to detect vehicles in individual video frames.

The system recognizes:

```text
Car
Motorcycle
Bus
Truck
```

Bounding boxes are drawn around detected vehicles.

---

### 6.3 Vehicle Tracking Module

Detected vehicles are assigned tracking IDs.

Tracking allows the system to determine how a vehicle moves between consecutive frames.

---

### 6.4 Speed Estimation Module

The distance travelled by a tracked vehicle between frames is calculated.

The system uses:

```text
Vehicle movement
        +
Video FPS
        ↓
Estimated speed
```

The current implementation uses a pixel-based scaling factor. Real-world calibration can be added as a future enhancement.

---

### 6.5 Congestion Analysis Module

The congestion algorithm uses two factors:

```text
Vehicle Density
       +
Vehicle Speed
       ↓
Congestion Score
```

The density contributes 60% and speed contributes 40%.

```python
congestion_score = (
    0.6 * density_score
    + 0.4 * speed_score
)
```

---

### 6.6 Alert Module

The alert module converts the congestion level into a user-friendly traffic message.

```text
LOW     → Normal traffic
MEDIUM  → Moderate traffic
HIGH    → High traffic
SEVERE  → Severe traffic
```

---

### 6.7 Dashboard Module

The Streamlit dashboard displays:

* Summary cards
* Congestion gauge
* Vehicle statistics
* Traffic charts
* Traffic alerts
* Processed video
* Session history
* CSV export

---

### 6.8 History Module

Completed analysis sessions are stored during the current Streamlit session.

The history includes:

* Time
* Total vehicles
* Cars
* Motorcycles
* Buses
* Trucks
* Average speed
* Congestion score
* Congestion level

---

## 7. Algorithm

### Step 1

Read the input traffic video.

### Step 2

Extract individual frames.

### Step 3

Run YOLO detection.

### Step 4

Filter detections to vehicle classes.

### Step 5

Track detected vehicles.

### Step 6

Calculate vehicle counts.

```text
Total =
Cars + Motorcycles + Buses + Trucks
```

### Step 7

Estimate speed using tracked vehicle movement.

### Step 8

Calculate density score.

```python
density_score = min(total / 30, 1.0)
```

### Step 9

Calculate speed score.

```python
speed_score = max(
    0,
    min((30 - average_speed) / 30, 1.0)
)
```

### Step 10

Calculate final congestion score.

```python
congestion_score = (
    0.6 * density_score
    + 0.4 * speed_score
)
```

### Step 11

Classify the congestion level.

```text
Score < 0.25       → LOW
Score < 0.50       → MEDIUM
Score < 0.75       → HIGH
Otherwise          → SEVERE
```

### Step 12

Generate the appropriate traffic alert.

### Step 13

Store the frame information.

### Step 14

Generate the processed video.

### Step 15

Display the results in Streamlit.

---

## 8. Expected Output

The system produces:

* Annotated traffic video
* Vehicle counts
* Average speed
* Congestion score
* Congestion level
* Traffic alert
* Vehicle history
* Congestion history
* Dashboard visualizations
* Downloadable traffic report
* CSV session history

---

## 9. Advantages

* Automated traffic analysis
* Reduces dependence on manual monitoring
* Provides visual analytics
* Considers both density and speed
* Provides immediate congestion classification
* Supports multiple vehicle categories
* Maintains analysis history
* Provides downloadable results
* Easy-to-use Streamlit interface

---

## 10. Limitations

The current system is primarily designed for video-based analysis.

The main limitations are:

* Speed is currently estimated rather than fully calibrated to real-world distance.
* Detection can be affected by video quality.
* Occlusion can affect tracking.
* Processing large videos can require significant computational resources.
* Direct live camera integration is not yet part of the current implementation.

---

## 11. Future Scope

Future development can include:

### Real-World Speed Calibration

Use known road distances or camera calibration to convert pixel movement into actual distance.

### Live Camera Monitoring

Connect the system to:

* CCTV
* Webcam
* RTSP camera streams

### Lane-Level Analysis

Analyze individual lanes separately.

### Incident Detection

Detect events such as:

* Stopped vehicles
* Abnormal traffic patterns
* Potential accidents

### Database Integration

Store historical traffic data permanently.

### Traffic Prediction

Use historical traffic data to predict future congestion.

### Cloud Deployment

Deploy the Streamlit application to a suitable cloud platform.

---

## 12. Conclusion

The Real-Time Traffic Congestion Analyzer demonstrates how computer vision and AI can be applied to traffic monitoring.

The system combines YOLO-based vehicle detection and tracking with speed estimation and a congestion algorithm based on both vehicle density and speed.

The Streamlit dashboard provides an accessible interface for viewing traffic statistics, congestion levels, alerts, charts, processed videos, and analysis history.

The project provides a foundation that can be extended toward calibrated speed measurement, live camera monitoring, lane-level analysis, historical traffic databases, and traffic prediction.
