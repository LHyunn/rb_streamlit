import streamlit as st
import os
from natsort import natsorted
import cv2
import numpy as np
from modules.preprocess import preprocessing as pp

"""
## Train Model. (In progress)

"""

col1, col2 = st.columns(2)
col1.write("Model Setting")
model_chkpt = col1.code("Checkpoint : single_channel_single_validation_CNN_weld_only_normalization_CNN.h5", language="json")
target_size = col1.selectbox("Target Size", options=["[512, 512, 1]", "[1024, 1024, 1]"])
weld_type = col1.selectbox("Weld Type", options=["True", "False"])
if weld_type == "False":
    preprocessing = col1.selectbox("Preprocessing", options=["None", "Histogram Equalization", "CLAHE", "Normalize"])
else:
    preprocessing = None
files = col1.file_uploader("Input Images or zip file. over 300 images", accept_multiple_files=True)

col2.write("Summary")
code = f"""
Model Checkpoint : {"single_channel_single_validation_CNN_weld_only_normalization_CNN"}
Target Size : {target_size}
Weld Type : {weld_type}
Preprocessing : {preprocessing}
"""
col2.code(code, language="json")

if files is not None:
    #save images
    save_img_path = "/app/data/User/image"
    save_zip_path = "/app/data/User/zip"
    
    for file in files:
        if file.type == "image/jpeg":
            with open(os.path.join(save_img_path, file.name), "wb") as f:
                f.write(file.getbuffer())
        elif file.type == "application/zip":
            with open(os.path.join(save_zip_path, file.name), "wb") as f:
                f.write(file.getbuffer())
            
            #unzip
            os.system(f"unzip {os.path.join(save_zip_path, file.name)} -d {save_img_path}")
            
    #load images
    img_list = natsorted(os.listdir(save_img_path))
            
            
    
