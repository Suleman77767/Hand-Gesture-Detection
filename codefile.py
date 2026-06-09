import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

st.title("🚗 Advanced Number Plate Detection (ANPR)")

st.write("Upload video and search vehicle number plate")

# Input
target_plate = st.text_input("Enter Plate Number to Search (e.g. ABC123)")

video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Load models
@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")  # general model (can be replaced with custom plate model)
    reader = easyocr.Reader(['en'])
    return yolo, reader

yolo_model, ocr_reader = load_models()

def read_plate(crop):
    result = ocr_reader.readtext(crop)
    text = ""

    for r in result:
        text += r[1] + " "

    return text.strip()

def detect_plate(frame):
    results = yolo_model(frame)

    plates = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            text = read_plate(crop)

            plates.append((text, (x1, y1, x2, y2)))

    return plates

if video_file:

    file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
    video = cv2.VideoCapture(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

    stframe = st.empty()

    found = False

    while True:
        ret, frame = video.read()
        if not ret:
            break

        plates = detect_plate(frame)

        for text, (x1, y1, x2, y2) in plates:

            # draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, text[:15], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            # match search
            if target_plate and target_plate.lower() in text.lower():
                found = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "MATCH FOUND", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

    if found:
        st.success("Vehicle Found in Video!")
    else:
        st.warning("No matching plate detected.")
