import streamlit as st
from yolo_predictions import YOLO_Pred
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide')

st.title('YOLO Object Detection for Image')
st.write('Please Uplaod Image to get detections')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./Model/weights/best.onnx',
                    data_yaml='./Model/data.yaml')

def upload_image():   
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize":"{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        #validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg)')
            return {"file":image_file,
                    "details": file_details}
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg,jpeg')
            return None

def main():
    object = upload_image()
    if object:
        prediction = False
        image_obj = Image.open(object['file'])
    
        col1, col2 = st.columns(2)
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)

        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detction from YOLO')
            if button:
                with st.spinner("""
                Getting Objects from image. please wait...."""):
                    # Resize the image to 224x224 pixels
                    resized_image = image_obj.resize((224, 224))  
                    image_array = np.array(resized_image)
                    pred_img = yolo.predictions(image_array)  
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True
    
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from Yolo V5 model")
            pred_img_obj_resize = pred_img_obj.resize((600, 600))
            st.image(pred_img_obj_resize)

if __name__=="__main__":
    main()






