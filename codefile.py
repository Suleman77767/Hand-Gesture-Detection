import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

st.set_page_config(page_title="ANPR System", layout="centered")

st.title("🚗 License Plate Detection System (Stable Version)")

# Input plate
target_plate = st.text_input("Enter Number Plate to Search (e.g. ABC123)")

# Upload video
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# OCR function
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise reduction
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # threshold for better OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(thresh)

    return text.strip()

# simple "plate region simulation"
def find_plate_regions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # filter likely plate region
        if w > 80 and h > 20 and w < 300 and h < 120:
            crop = frame[y:y+h, x:x+w]
            plates.append((crop, (x, y, w, h)))

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

        plates = find_plate_regions(frame)

        detected_text = ""

        for crop, (x, y, w, h) in plates:

            text = extract_text(crop)

            detected_text += text + " "

            # draw box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, text[:15], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # match search
            if target_plate and target_plate.lower() in text.lower():
                found = True

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.putText(frame, "MATCH FOUND", (x, y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

        stframe.image(frame, channels="BGR")

    if found:
        st.success("🚗 Vehicle Found in Video!")
    else:
        st.warning("No matching number plate found.")
