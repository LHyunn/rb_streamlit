import os

import streamlit as st
import shutil
import sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from natsort import natsorted
import cv2

from glob import glob

"""
## Result.

"""
def get_pred(df, image):
    pred = df[df["path"] == image]["predict"].values[0]
    return pred

col1, col2, col3 = st.columns(3)

test_result_df = pd.read_csv("/app/temp/test_result.csv")
test_result_df = test_result_df.sort_values(by="path")
test_result_df = test_result_df.reset_index(drop=True)
col1.dataframe(test_result_df, height=500)

try:
    hit_ratio_df = pd.read_csv("/app/temp/hit_evaluation_result.csv")
    col2.dataframe(hit_ratio_df, height=500)
    col3.image("/app/temp/hit_ratio.png")
except:
    col2.write("hit_evaluation_result.csv 파일이 없습니다.")
    col3.write("hit_ratio.png 파일이 없습니다. 레이블이 없는 데이터셋인 경우 생성되지 않습니다.")



col_1, col_2 = st.columns(2)

images = os.listdir(f"/app/temp/CAM")
images = natsorted(images)
images_list = []
for i in range(len(images)):
    image = cv2.imread(f"/app/temp/CAM/{images[i]}", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_list.append(image)
    
for image, i in zip(images, range(len(images_list))):
    if i % 2 == 0:
        col_1.write(f"이미지 : {image}")
        col_1.write(f"예측 : {get_pred(test_result_df, image)}")
        col_1.image(images_list[i])
    elif i % 2 == 1:
        col_2.write(f"이미지 : {image}")
        col_2.write(f"예측 : {get_pred(test_result_df, image)}")
        col_2.image(images_list[i])