import cv2
import numpy as np
from glob import glob
import modules.preprocess.preprocessing as preprocessing

def load_test(data_path, target_size, weld_type = False, preprocess = None):
    """테스트 데이터를 불러오는 함수
    """
    if weld_type == True and preprocess is not None:
        raise ValueError("weld_type과 preprocess를 동시에 사용할 수 없습니다.")
    test_image = glob(data_path + "/**/*.png")
    test_image_label = [int(i.split("/")[-2]) for i in test_image]
    test_image_list = []
    for i in test_image:
        img = cv2.imread(i, (cv2.IMREAD_GRAYSCALE if target_size[2] == 1 else cv2.IMREAD_COLOR))
        img = cv2.resize(img, (target_size[0], target_size[1]))
        if weld_type:
            img = preprocessing.get_weld_image(img)
            
        if preprocess is not None:
            img = preprocessing.preprocess_img(img, preprocess, target_size)
            
        img = np.array(img, dtype=np.float32)
        img = img / 255.
            
        
        test_image_list.append(img)
    test_image_list = np.array(test_image_list)
    test_image_list = test_image_list.reshape(-1, target_size[0], target_size[1], target_size[2])
    return test_image_list, test_image_label, test_image