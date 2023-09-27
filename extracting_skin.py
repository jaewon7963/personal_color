import numpy as np
import cv2 as cv
import os

from PIL import Image
from sklearn.cluster import KMeans

input_dir = './hyegyo'
output_dir = './spring_warm_light3'

face_cascade = cv.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')


"""
논문에서 제안하는 threshold
  Y       Cr      Cb
0-255, 133-173, 80-120


lower = np.array([0,143,82], dtype = np.uint8)
upper = np.array([255,163,117], dtype = np.uint8)


"""
lower = np.array([0,143,70], dtype = np.uint8)
upper = np.array([255,173,125], dtype = np.uint8)

"""
이미지에서 가장 많이 나오는 색을 추출해 팔레트에 표시해주는 함수
"""
def palette(clusters):
    width = 300
    palette = np.zeros((50,width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx , centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette


def facedetect(image):
    count = 0
    try:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        """
        haar로 얼굴 검출하기
        """
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            count = count + 1
        
        if count != 1:
            os.remove(image)
            return None
         
        """
        검출한 좌표로 얼굴만 추출하기
        """
        crop_img = img[y:y+h, x:x+w]
        ycbcr_img = cv.cvtColor(crop_img, cv.COLOR_BGR2YCR_CB)
        
        
        """
        피부 검출을 위한 마스크 생성
        """
        skin_mask = cv.inRange(ycbcr_img, lower, upper)
        
        
        """
        마스크를 이용한 AND로 얼굴에서 피부만 추출
        """
        skin = cv.bitwise_and(ycbcr_img, ycbcr_img, mask = skin_mask)
        rgb_skin = cv.cvtColor(skin, cv.COLOR_YCR_CB2BGR)
        
        return rgb_skin
    
    
    except Exception as e:
        print(f"예외 발생: {e}")

        os.remove(image)
        return None
        
def skin_extract(skin):
    clt_1 = KMeans(n_clusters=2)
    clt_1.fit(skin.reshape(-1,3))

    palette_img = palette(clt_1)

    skin_color = (0,0,0)

    lab_palette = cv.cvtColor(palette_img, cv.COLOR_BGR2LAB)

    """
    팔레트 영역에서 배경과 가장 많이 나온 피부색 중, 피부색 선택하기
    """


    max_area = -1
    # 5개의 후보군 중 R 값이 가장 높은 피부색 선택
    for row in palette_img:
        for color in row:
            if max_area < color[2]:
                max_area = color[2]
                skin_color = color
    
    """
    for row in palette_img:
        for color in row:
            if color[2] >= 70:
                skin_color = color
                break
            else:
                continue
        break
    """

    if (np.array(skin_color) == np.array([0, 0, 0])).all():
        return None

    rgb_skin_color = (skin_color[2], skin_color[1], skin_color[0])
    image = Image.new("RGB", (50, 50), rgb_skin_color)
    
    return image





for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        
        print(filename)
        
        rgb_skin = facedetect(input_dir + "/" + filename)
        if rgb_skin is None:
            continue
        image = skin_extract(rgb_skin)
        if image is None:
            continue

        output_path = os.path.join(output_dir, filename.split('.')[0] + '.png')
        image.save(output_path)
