import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from yolo_predictions import YOLO_Pred
import tempfile
import os
import cv2

# Load YOLO model
yolo = YOLO_Pred(onnx_model='./Model/weights/best.onnx',
                 data_yaml='./Model/data.yaml')

def process_video_file(video_file):
    # Read the video file
    #container = av.open('./images/test.mp4')
    container = av.open(video_file)

    # Create a temporary directory to store processed frames
    temp_dir_path = './images/temp'

    # Iterate through video frames
    for i, frame in enumerate(container.decode(video=0)):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        # Perform YOLO detection
        pred_img = yolo.predictions(img)

        # Save the processed frame
        cv2.imwrite(os.path.join(temp_dir_path, f"frame_{i:04d}.jpg"), pred_img)

    return temp_dir_path

def main():
    st.title("YOLO Video Object Detection")

    # File uploader for video
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        # Display the uploaded video
        st.video(uploaded_file)

        # Process the uploaded video file
        st.write("Performing object detection...")
        with st.spinner('Processing...'):
            temp_dir_path = process_video_file(uploaded_file)

        # Display the processed video
        st.write("Object detection completed!")
        st.write("Displaying detected video...")
        for filename in sorted(os.listdir(temp_dir_path)):
            st.image(os.path.join(temp_dir_path, filename))

        
if __name__ == "__main__":
    main()
