import os

import streamlit as st
import shutil
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob

from natsort import natsorted
from contextlib import redirect_stdout

from modules.preprocess import preprocessing as pp
from modules.test import test as tt
from modules.inference import inference as inf
"""
## Model Inference. (In progress)

"""

with st.expander("Explanation"):
    st.write("""
             
            샘플 데이터 경로는 관리자에게 문의 바랍니다.
                        
            """)
    


########################################################################################################################
message = st.empty()
model_path = st.selectbox("Select Model", options=glob("/app/models/**/*.h5", recursive=True))
message.info("Please wait for loading model...")
model = tf.keras.models.load_model(model_path)
message.success("Model loaded successfully. Ready to inference.")
########################################################################################################################
col1, col2 = st.columns(2)
col1.write("Model Setting")
model_chkpt = col1.code(f'Checkpoint : {model_path.split("/")[-1]}', language="json")
last_conv_layer = col1.selectbox("Last Conv Layer", options=[layer.name for layer in model.layers])
target_size = col1.selectbox("Target Size", options=["[512, 512, 1]", "[1024, 1024, 1]"])
weld_type = col1.selectbox("Weld Type", options=["True", "False"])
sample_data_path = col1.text_input("Sample Data Path", value="/app/data/HyundaiRB/Data/test")
########################################################################################################################
if target_size == "[512, 512, 1]":
    target_size = [512, 512, 1]
else:
    target_size = [1024, 1024, 1]
if weld_type == "False":
    weld_type = False
    preprocessing = col1.selectbox("Preprocessing", options=["None", "Histogram Equalization", "CLAHE", "Normalize"])
else:
    preprocessing = None
    weld_type = True
########################################################################################################################
col2.write("Summary")
code = f"""
Model Checkpoint : {model_path.split("/")[-1]}
Last Conv Layer : {last_conv_layer}
Target Size : {target_size}
Weld Type : {weld_type}
Preprocessing : {preprocessing}
"""
col2.code(code, language="json")
########################################################################################################################
summary_str = None
with open('summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
with open('summary.txt', 'r') as f:
    summary_str = f.read()
os.remove('summary.txt')
with col2.expander("Model Summary"):
    st.code(summary_str, language="json")
########################################################################################################################

########################################################################################################################
col1_1, col1_2, col1_3, col1_4, col1_5, col1_6, col1_7, col1_8 = st.columns(8)
if col1_8.button("Inference"):
    try:
        shutil.rmtree("/app/temp")
    except:
        pass
    finally:
        os.makedirs("/app/temp/CAM", exist_ok=True)
    images_count = len(glob(f"{sample_data_path}/**/*.png", recursive=True))
    if images_count != 0:
        message.info("Inference Start. This may take a while...")
        test_image_list, test_image_label, test_image_path = tt.load_test(sample_data_path, target_size, weld_type = weld_type, comp=message)
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer).output, model.output])
        message.info("Loaded data, Total images : {}".format(images_count))
        time.sleep(1)
        message.info("Predicting...")
        col1, col2 = st.columns(2)
        if test_image_label is None:
            tt.test_model_no_label(model, test_image_list, test_image_path)
        else:
            tt.test_model(model, test_image_list, test_image_label, test_image_path)
        message.info("Predicting... Done.")
        time.sleep(1)
        message.info("Making Grad-CAM...")
        for image, path, i in zip(test_image_list, test_image_path, range(len(test_image_list))):
            image_exp = np.expand_dims(image, axis=0)
            file_name = path.split("/")[-1].split(".")[0]
            heat_map, pred = inf.make_gradcam_heatmap(image_exp, grad_model)
            inf.save_and_display_gradcam(image, heat_map, f"/app/temp/CAM/{file_name}.png")
            message.info("Making Grad-CAM... {}/{}".format(i+1, len(test_image_list)))
        message.info("Making Grad-CAM... Done.")
        time.sleep(1)
        message.success("Inference Done. Please check the result.")
        
    else:
        message.error("There is no image in the directory.")
    
    
        
    
    



            
            
    
