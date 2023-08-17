import streamlit as st
import os
from natsort import natsorted
import cv2
from glob import glob
"""
## Image Preview

### 모든 데이터는 외부로의 유출을 금합니다.

"""

col1, col2, col3, col4, col5 = st.columns(5)
NDT_result = col1.selectbox("전체 / 불량", natsorted(os.listdir("/app/data/HyundaiRB/Data/Raw_Data_jpg")))
thickness = col2.selectbox("두께", natsorted(os.listdir(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}")))
pipe = col3.selectbox("강관", natsorted(os.listdir(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}")))
col5.write(" ")
col5.write(" ")


col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
path = col1.text_input("또는 경로를 입력하여 해당 경로 내의 이미지를 모두 조회", f"/{NDT_result}/{thickness}/{pipe}")
limit = col2.number_input("조회할 이미지 수 상한", min_value=1, max_value=500, value=100, step=1)
preprocess2 = col3.selectbox("전처리", ["원본", "히스토그램 평활화", "CLAHE", "Normalize"], key="preprocess2")
col4.write(" ")
col4.write(" ")
button2 = col4.button("조회", key="button2")


col_1, col_2, col_3, col_4, col_5, col_6, col_7 = st.columns(7)
if button2:
    if path.endswith("/"):
        path = path[:-1]

    path1 = f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}"
    path2 = f"/app/data/HyundaiRB/Data/Raw_Data_jpg" + path
    
    if path1 != path2:
        images = glob("/app/data/HyundaiRB/Data/Raw_Data_jpg" + path + "/**/*.jpg", recursive=True)
    else:
        images = glob(f"/app/data/HyundaiRB/Data/Raw_Data_jpg/{NDT_result}/{thickness}/{pipe}/*.jpg")
    images = natsorted(images)
    images = images[:limit]
    images_list = []
    if preprocess2 == "원본":
        for i in range(len(images)):
            image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            images_list.append(image)
            
    elif preprocess2 == "히스토그램 평활화":
        for i in range(len(images)):
            image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            image = cv2.equalizeHist(image)
            images_list.append(image)
            
    elif preprocess2 == "CLAHE":
        for i in range(len(images)):
            image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
            images_list.append(image)
            
    elif preprocess2 == "Normalize":
        for i in range(len(images)):
            image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            images_list.append(image)
            
    elif preprocess2 == "흑백 반전":
        for i in range(len(images)):
            image = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            image = 255 - image
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
   