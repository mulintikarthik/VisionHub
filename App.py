import cv2
import streamlit as st
import numpy as np

# Load Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier(r'C:\Users\reddy\Downloads\Projects\VisionHub\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\reddy\Downloads\Projects\VisionHub\haarcascade_eye (1).xml')

# Streamlit UI
st.set_page_config(page_title="VisionHub - Face & Eye Detection", layout="wide")
st.title("👁️ VisionHub - Real-Time Face & Eye Detection App")
st.write("Detect Faces and Eyes using OpenCV Haar Cascade Classifiers!")

menu = ["Home", "Image Detection", "Webcam Detection"]
choice = st.sidebar.selectbox("Select Mode", menu)

def detect_faces_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img

if choice == "Home":
    st.subheader("Welcome to VisionHub 🚀")
    st.markdown("""
        - **Image Detection**: Upload an image to detect faces and eyes.  
        - **Webcam Detection**: Perform real-time detection using your webcam.  
    """)

elif choice == "Image Detection":
    st.subheader("📸 Image Detection")
    upload_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if upload_file is not None:
        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        result_img = detect_faces_eyes(img)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption='Processed Image')

elif choice == "Webcam Detection":
    st.subheader("📷 Webcam Real-Time Detection")
    run = st.checkbox("Start Camera")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_faces_eyes(frame)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
