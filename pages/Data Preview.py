import streamlit as st
import os
from natsort import natsorted
import cv2
"""
## Image Preview

### 모든 데이터는 외부로의 유출을 금합니다.

"""

col1, col2, col3, col4, col5 = st.columns(5)
NDT_result = col1.selectbox("전체 / 불량", natsorted(os.listdir("/app/data/HyundaiRB/Data/Raw_Data_jpg")))
thickness = col2.selectbox("두께", natsorted(os.listdir(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}")))
pipe = col3.selectbox("강관", natsorted(os.listdir(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}")))
preprocess = col4.selectbox("전처리", ["원본", "히스토그램 평활화", "CLAHE", "Normalize"])
col5.write(" ")
col5.write(" ")
button = col5.button("조회")


col_1, col_2, col_3, col_4, col_5, col_6, col_7 = st.columns(7)
if button:
    images = os.listdir(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}")
    images = natsorted(images)
    images_list = []
    if preprocess == "원본":
        for i in range(len(images)):
            image = cv2.imread(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}/{images[i]}", cv2.IMREAD_GRAYSCALE)
            images_list.append(image)
    
    elif preprocess == "히스토그램 평활화":
        for i in range(len(images)):
            image = cv2.imread(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}/{images[i]}", cv2.IMREAD_GRAYSCALE)
            image = cv2.equalizeHist(image)
            images_list.append(image)
            
    elif preprocess == "CLAHE":
        for i in range(len(images)):
            image = cv2.imread(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}/{images[i]}", cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            images_list.append(image)
            
    elif preprocess == "Normalize":
        for i in range(len(images)):
            image = cv2.imread(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}/{images[i]}", cv2.IMREAD_GRAYSCALE)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            images_list.append(image)
            
            
            
            
    
    for image, i in zip(images, range(len(images_list))):
        if i % 7 == 0:
            col_1.image(images_list[i])
        elif i % 7 == 1:
            col_2.image(images_list[i])
        elif i % 7 == 2:
            col_3.image(images_list[i])
        elif i % 7 == 3:
            col_4.image(images_list[i])
        elif i % 7 == 4:
            col_5.image(images_list[i])
        elif i % 7 == 5:
            col_6.image(images_list[i])
        elif i % 7 == 6:
            col_7.image(images_list[i])
            
            
