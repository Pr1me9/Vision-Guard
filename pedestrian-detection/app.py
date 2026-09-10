import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO
import tempfile
import threading

# Load YOLOv9 model
model = YOLO('./yolov9c.pt')

# Constants
human_class = 0
vehicle_classes = [1, 2, 3, 5, 7]
KNOWN_WIDTH = 0.5  # meters
FOCAL_LENGTH = 700  # pixels

# Optical Flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Globals
prev_frame = None
prev_points = None
frame_count = 0
detections_summary = {
    "total_detections": 0,
    "vehicles_detected": 0,
    "pedestrians_detected": 0
}


def estimate_distance(box):
    box_width = box[2] - box[0]
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
    return distance


def is_near(distance, threshold=4.0):
    return distance < threshold


def send_alert():
    st.warning("🚨 ALERT: Nearby pedestrian detected!")


def track_objects(frame, detections):
    global prev_frame, prev_points

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray_frame

    if prev_frame.shape != gray_frame.shape:
        gray_frame = cv2.resize(gray_frame, (prev_frame.shape[1], prev_frame.shape[0]))

    if detections:
        new_points = []
        for result in detections.boxes:
            box = result.xyxy[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())
            if cls in [human_class] + vehicle_classes:
                new_points.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if new_points:
            prev_points = np.float32(new_points).reshape(-1, 1, 2)

    try:
        if prev_points is not None and prev_points.size > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, prev_points, None, **lk_params)
            for i, (new, old) in enumerate(zip(next_points, prev_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                speed = np.sqrt((a - c) ** 2 + (b - d) ** 2)
                direction = np.arctan2(b - d, a - c) * 180 / np.pi
                cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.putText(frame, f'Speed: {speed:.2f} Dir: {direction:.2f}',
                            (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            prev_points = next_points
    except cv2.error as e:
        print(f"Optical flow error: {e}")
        prev_points = None

    prev_frame = gray_frame
    return frame


def detect_and_alert(frame):
    global detections_summary, frame_count

    if frame_count % 1 == 0:
        results = model.predict(frame)[0]
        alerts = []

        frame = track_objects(frame, results)

        for result in results.boxes:
            box = result.xyxy[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())
            conf = result.conf[0].cpu().numpy()

            if cls == human_class:
                distance = estimate_distance(box)
                if is_near(distance):
                    alerts.append(box)
                label = model.names[int(cls)]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f} Dist: {distance:.2f}m',
                            (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                detections_summary["pedestrians_detected"] += 1
            elif cls in vehicle_classes:
                label = model.names[int(cls)]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                detections_summary["vehicles_detected"] += 1

        detections_summary["total_detections"] = detections_summary["pedestrians_detected"] + \
                                                 detections_summary["vehicles_detected"]

        if alerts:
            threading.Thread(target=send_alert).start()

    frame_count += 1
    return frame


def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = detect_and_alert(frame)
    return frame


def process_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detect_and_alert(frame)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frames.append(processed_frame)

    cap.release()
    return frames


# Streamlit UI
# Center-aligned big title in ALL CAPS with emojis
st.markdown(
    """
    <h1 style='text-align: center; font-weight: bold;'>
        👁️‍🗨️🛡️ VISIONGUARD 🛡️👁️‍🗨️
    </h1>
    """,
    unsafe_allow_html=True
)

# Smaller subtitle below
st.title("🚦 Real-Time Pedestrian & Vehicle Detection (YOLOv9 + Optical Flow)")
option = st.radio("Choose an option", ["Upload Image", "Upload Video", "Camera (Live)"], index=0)

if option == "Upload Image":
    image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image_file:
        with st.spinner("Processing..."):
            output_frame = process_image(image_file)
            st.image(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)

elif option == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        with st.spinner("Processing video..."):
            frames = process_video(video_file)
            for f in frames:
                st.image(f)
            st.success("Video processing complete!")

elif option == "Camera (Live)":
    st.info("Live camera streaming not supported in Streamlit natively. Try running via local webcam capture loop if needed.")

# Stats
st.markdown("## 📊 Detection Summary")
st.json(detections_summary)
