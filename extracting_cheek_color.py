import numpy as np
import cv2 as cv
import os

from collections import Counter
from PIL import Image
from sklearn.cluster import KMeans

gt_list = ['spring_warm_bright' , 'spring_warm_light', 'autumn_warm_mute', 'autumn_warm_deep',
           'summer_cool_mute', 'summer_cool_light', 'winter_cool_bright', 'winter_cool_deep']

name = "winter"
input_dir = './result/chin/' + name
output_dir = './result/chin' + "/" + name + "_result"

face_cascade = cv.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


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


def skin_extract(file):
    
    skin = cv.imread(file)
    clt = KMeans(n_clusters=5)
    clt_1 = clt.fit(skin.reshape(-1, 3))


    color_list = []
    most_common_array = []
    color_scalar_cheek = (0,0,0)

    """
    후보군 색상 팔레트가 없다면 예외발생
    """
    
    try:
        palette_img = palette(clt_1)
    except Exception as e:
        print(f"Error: {e}")
        return None

    skin_color = (0,0,0) 

    try:
        for row in palette_img:
            for color in row:
                color_list.append(color)
                
        filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 128, 0])) and arr[2] >= 90\
                               and arr[2] >= arr[1] and arr[2] >= arr[0]]
        freq = Counter(map(tuple, filtered_array_list))
                
        most_common_array, most_common_count = freq.most_common(1)[0]
        
        #print("----------------- 팔레트 빈도수")
        #print(f"배열 빈도수: {freq}")
        #print(f"가장 빈도수가 큰 배열: {np.array(most_common_array)}")
        #print(f"빈도수: {most_common_count}")
        #print("----------------------------")
    #color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
        color_scalar_cheek = (int(most_common_array[0]),int(most_common_array[1]) , int(most_common_array[2]))
        
    except Exception as e:
        print("피부를 검출할 수 없는 이미지")
        return None
    
    skin_color = color_scalar_cheek
    
    """
    피부를 검출하지 못한 경우 제외
    """
    if (np.array(skin_color) == np.array([0, 0, 0])).all():
        return None

    if skin_color[2] < 65:
        return None

    rgb_skin_color = (skin_color[2], skin_color[1], skin_color[0])
    image = Image.new("RGB", (50, 50), rgb_skin_color)
    
    return image



for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        
        print(filename)
        
        image = skin_extract(input_dir + "/" + filename)
        if image is None:
            continue

        output_path = os.path.join(output_dir, filename.split('.')[0] + '.png')
        image.save(output_path)