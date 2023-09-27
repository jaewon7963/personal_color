import numpy as np
import cv2 as cv
import os

from collections import Counter
from PIL import Image
from sklearn.cluster import KMeans

gt_list = ['spring_warm_bright' , 'spring_warm_light', 'autumn_warm_mute', 'autumn_warm_deep',
           'summer_cool_mute', 'summer_cool_light', 'winter_cool_bright', 'winter_cool_deep']

name = "cool_s_contour_result"
input_dir = './result/eye/' + name
output_dir = './result/eye' + "/" + name + "_result"

face_cascade = cv.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

"""
논문에서 제안하는 threshold
  Y       Cr      Cb
0-255, 133-173, 80-120


lower = np.array([0,143,82], dtype = np.uint8)
upper = np.array([255,163,117], dtype = np.uint8)


"""
lower = np.array([255,255,255], dtype = np.uint8)
upper = np.array([255,255,255], dtype = np.uint8)


lower_otsu_e = np.array([0,0,0], dtype = np.uint8)
upper_otsu_e = np.array([0,0,0], dtype = np.uint8)


"""
lowerb = np.array([0, 35, 80], dtype = "uint8")
upperb = np.array([20, 255, 255], dtype = "uint8")
"""

lowerb = np.array([0, 34, 80], dtype = "uint8")
upperb = np.array([20, 200, 255], dtype = "uint8")

"""
이미지에서 가장 많이 나오는 색을 추출해 팔레트에 표시해주는 함수
"""
def palette(k_cluster):
    width = 300
    palette = np.zeros((50,width, 3), np.uint8)
    
    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)
    perc = {}
    
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
        
    perc = dict(sorted(perc.items()))
    
    step = 0
    
    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)

    return palette

count = 0


def facedetect(image):
    try:
        
        img = cv.imread(image)
        
        haar_eye_size = cv.resize(img, dsize = (70, 70), interpolation=cv.INTER_LINEAR)

        
        """
        otsu 를 이용하여 머리카락 등등 제거
        """
        t, t_otsu = cv.threshold(cv.cvtColor(haar_eye_size, cv.COLOR_BGR2GRAY), -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        bgr_otsu = cv.cvtColor(t_otsu, cv.COLOR_GRAY2BGR)
        
        
        """
        피부 검출을 위한 마스크 생성, otsu
        """
        
        eye_only = cv.inRange(bgr_otsu, lower_otsu_e, upper_otsu_e)
        
        """
        마스크를 이용한 AND로 얼굴에서 피부만 추출
        """

        eye_otsu_skin = cv.bitwise_and(haar_eye_size, haar_eye_size, mask = eye_only)
        
        return eye_otsu_skin
    
    
    except Exception as e:
        print(f"예외 발생: {e}")

        #os.remove(image)
        return None
        
def skin_extract(skin):
    clt = KMeans(n_clusters=5)
    clt_1 = clt.fit(skin.reshape(-1, 3))


    color_list = []
    most_common_array = []
    color_scalar_eye = (0,0,0)

    """
    후보군 색상 팔레트가 없다면 예외발생
    """
    
    try:
        palette_img = palette(clt_1)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    """
    cv.imshow('eye palette', palette_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    
    skin_color = (0,0,0)

    """
    # 5개의 후보군 중 배경을 제외하고 R 값이 가장 높은 피부색(선명한 색) 선택
    """
    max_area = -1
    """
    for row in palette_img:
        for color in row:
            if max_area < color[2] and \
                abs(color[2] - color[1]) >= 30 and abs(color[2] - color[0]) >= 30 and \
                    color[2] > color[1] and color[2] > color[1]:
                max_area = color[2]
                skin_color = color
    """
    

    try:
        for row in palette_img:
            for color in row:
                color_list.append(color)
                
        filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0])) \
                               and arr[2] <=65 ]
        freq = Counter(map(tuple, filtered_array_list))
                
        most_common_array, most_common_count = freq.most_common(1)[0]
        
        #print("----------------- 팔레트 빈도수")
        #print(f"배열 빈도수: {freq}")
        #print(f"가장 빈도수가 큰 배열: {np.array(most_common_array)}")
        #print(f"빈도수: {most_common_count}")
        #print("----------------------------")
    #color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
        color_scalar_eye = (int(most_common_array[0]),int(most_common_array[1]) , int(most_common_array[2]))
        
    except Exception as e:
        print("피부를 검출할 수 없는 이미지")
        return None
    
    skin_color = color_scalar_eye
    """
    
    for row in palette_img:
        for color in row:
            if max_area < color[2]:
                max_area = color[2]
                skin_color = color    
                
                
             
    color_list = []

    for row in palette_img:
        for color in row:
            color_list.append(color)
    """

    """
    try:
        filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0]))]
        freq = Counter(map(tuple, filtered_array_list))
                
        most_common_array, most_common_count = freq.most_common(1)[0]
        
        skin_color[2] = most_common_array[2]
        skin_color[1] = most_common_array[1]
        skin_color[0] = most_common_array[0]
    except Exception as er:
        os.remove(input_dir + "/" + filename)
        return None
    """

           
    """ # 팔레트 영역에서 배경과 가장 많이 나온 피부색 중, 피부색 선택하기
    for row in palette_img:
        for color in row:
            if color[2] >= 70:
                skin_color = color
                break
            else:
                continue
        break
    """

    """
    피부를 검출하지 못했거나, R이 너무 낮은 경우 제외
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