import numpy as np
import cv2 as cv
import os

from collections import Counter
from PIL import Image
from sklearn.cluster import KMeans

input_dir = './nut'
output_dir = './nut_result/'

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

def extract_most_common_color(palette):
    color_list = []
    for row in palette:
        for color in row:
            color_list.append(color)

    filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0])) and arr[2] >= 5]
    freq = Counter(map(tuple, filtered_array_list))

    most_common_array, most_common_count = freq.most_common(1)[0]

    return (int(most_common_array[0]), int(most_common_array[1]), int(most_common_array[2]))

""" # 확인용
image = cv.imread(input_dir + "/027.PNG")

clt = KMeans(n_clusters = 5)
clt_1 = clt.fit(image.reshape(-1,3))
palette_img = palette(clt_1)
color = (0,0,0)

try:
    color = extract_most_common_color(palette_img)
except:
    pass

cv.imshow('palette', palette_img)
cv.waitKey(0)
cv.destroyAllWindows()


color_img = Image.new("RGB", (50, 50), (color[2], color[1], color[0]))
color_img.show()
"""



for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.PNG'):
        
        print(filename)
        
        image = cv.imread(input_dir + "/" + filename)
        if image is None:
            continue
        
        clt = KMeans(n_clusters = 5)
        clt_1 = clt.fit(image.reshape(-1,3))
        palette_img = palette(clt_1)
        color = (0,0,0)
        
        try:
            color = extract_most_common_color(palette_img)
        except:
            pass
        
        color_img = Image.new("RGB", (50, 50), (color[2], color[1], color[0]))

        output_path = os.path.join(output_dir, filename.split('.')[0] + '_result.png')
        color_img.save(output_path)
