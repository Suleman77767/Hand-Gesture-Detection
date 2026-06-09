import streamlit as st
import cv2
import numpy as np
import easyocr

st.set_page_config(page_title="ANPR System", layout="centered")

st.title("🚗 License Plate Detection & Recognition System")

st.write("Upload video and search for a number plate")

# User input
target_plate = st.text_input("Enter Number Plate (e.g. ABC123)")

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Load EasyOCR once
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

reader = load_reader()

# OCR function
def read_text(frame):
    result = reader.readtext(frame)
    text = ""

    for r in result:
        text += r[1] + " "

    return text.strip()

if video_file:

    file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
    video = cv2.VideoCapture(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))

    stframe = st.empty()

    found = False

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # resize for speed
        frame = cv2.resize(frame, (720, 420))

        # OCR directly on frame (no contour detection)
        text = read_text(frame)

        # draw detected text
        cv2.putText(frame, text[:40], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        # match plate
        if target_plate and target_plate.lower() in text.lower():
            found = True

            cv2.rectangle(frame, (50, 50), (650, 350), (0, 0, 255), 3)
            cv2.putText(frame, "MATCH FOUND", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

        stframe.image(frame, channels="BGR")

    if found:
        st.success("🚗 Vehicle Found in Video!")
    else:
        st.warning("No matching number plate detected.")
