import numpy as np
import cv2 as cv
import rgb2hsv as r2h
import time
import pandas as pd

from catboost import CatBoostClassifier, Pool
from colormath.color_objects import sRGBColor, LabColor, HSVColor
from colormath.color_conversions import convert_color
from collections import Counter
from sklearn.cluster import KMeans

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

def palette2(clusters):
    width = 300
    palette = np.zeros((50,width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx , centers in enumerate(clusters.cluster_centers_):
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette

def extract_most_common_color(palette):
    color_list = []
    for row in palette:
        for color in row:
            color_list.append(color)

    filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 128, 0])) and arr[2] >= 65]
    freq = Counter(map(tuple, filtered_array_list))

    most_common_array, most_common_count = freq.most_common(1)[0]

    return (int(most_common_array[0]), int(most_common_array[1]), int(most_common_array[2]))


count = 0

"""
input_path = './image'
name = 'kalina'
num ='_017'
"""

"""
톤의 기본 값은 웜톤(False)
"""
tone = False 
weather = False
mood = False


"""
피부 톤 진단을 위한 기준
"""
thresh = 4.0

s_for_cool = 25.3
v_for_warm = 88
s_for_warm = 20.3
v_for_summer = 88.5
h_for_cool = 17.7

tone_list = ["봄 웜 브라이트" , "봄 웜 라이트", "여름 쿨 라이트", "여름 쿨 뮤트", "가을 웜 뮤트", "가을 웜 딥", "겨울 쿨 딥", "겨울 쿨 브라이트"]
count = 0



"""
논문에서 제안하는 threshold
  Y       Cr      Cb
0-255, 133-173, 80-120
"""
lower = np.array([0,123,75], dtype = np.uint8)
upper = np.array([255,173,135], dtype = np.uint8)

lower_otsu = np.array([255,255,255], dtype = np.uint8)
upper_otsu = np.array([255,255,255], dtype = np.uint8)

"""
초기 값은
[0, 48, 80]
[20, 255, 255] 로 진행하였음
"""

lowerb = np.array([0, 34, 80], dtype = "uint8")
upperb = np.array([20, 200, 255], dtype = "uint8")

face_cascade = cv.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
eye_detector = cv.CascadeClassifier('./xml/haarcascade_eye.xml')


"""
이미지 입력
"""
start_time = time.time()


#img = cv.imread('./image/person/003.jpg')
#img = cv.imread(input_path + "/" + name + "/" + name + num + '.jpg')
img = cv.imread('./static/test.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


"""
haar로 얼굴 검출하기
"""
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    count += 1
 
print(count)


"""

"""
#haar로 눈 검출하기
"""
max_len = 0
haar_eye = None

eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=3)
for (ex,ey,ew,eh) in eyes:
    if max_len < ey + eh:
        max_len = ey + eh
        haar_eye = roi_color[ey:ey + eh, ex:ex + ew]
    #cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

"""

"""
검출한 좌표로 얼굴만 추출하기
"""
crop_img = img[y:y+h, x:x+w]
crop_gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
hsv_img = cv.cvtColor(crop_img, cv.COLOR_BGR2HSV)
ycbcr_img = cv.cvtColor(crop_img, cv.COLOR_BGR2YCR_CB)


"""
otsu 를 이용하여 머리카락 등등 제거
"""
t, t_otsu = cv.threshold(crop_gray, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cvt_otsu = cv.cvtColor(t_otsu, cv.COLOR_GRAY2BGR)


"""
피부 검출을 위한 마스크 생성
"""
# Ycbcr 을 이용한 마스크
skin_mask = cv.inRange(ycbcr_img, lower, upper)
# otsu 를 이용한 마스크
only_skin = cv.inRange(cvt_otsu, lower_otsu, upper_otsu)

"""
마스크를 이용한 AND로 얼굴에서 피부만 추출
"""
skin = cv.bitwise_and(ycbcr_img, ycbcr_img, mask = skin_mask)
rgb_skin = cv.cvtColor(skin, cv.COLOR_YCR_CB2BGR)
#lab_skin = cv.cvtColor(skin, cv.COLOR_BGR2LAB)

# otsu 마스크로 AND 연산
skin_otsu = cv.bitwise_and(crop_img, crop_img, mask = only_skin)


"""
클러스터를 사용하기 위한 KMeans, n_clusters 의 개수는 팔레트에 표시되는 색상의 수
"""
clt = KMeans(n_clusters=5) # YCR skin
clt2 = KMeans(n_clusters=5) # otsu skin
clt3 = KMeans(n_clusters = 5) # chin
clt4 = KMeans(n_clusters = 5) # cheek
clt5 = KMeans(n_clusters = 5) # forehead
clt6 = KMeans(n_clusters = 5) # eye haar
#clt7 = KMeans(n_clusters = 5) # eye contour
#clt8 = KMeans(n_clusters = 5) # eye contour mask


clt_1 = clt.fit(rgb_skin.reshape(-1, 3))


#clt_2 = clt2.fit(skin_otsu.reshape(-1, 3))

"""
# hsv 색 분포로 피부 검출하기
"""
hsv_skin = cv.cvtColor(skin_otsu, cv.COLOR_BGR2HSV)
hsv_mask = cv.inRange(hsv_skin, lowerb, upperb)

hsv_skin_skin = cv.bitwise_and(hsv_skin, hsv_skin, mask = hsv_mask)
rgb_hsv_skin = cv.cvtColor(hsv_skin_skin, cv.COLOR_HSV2BGR)

"""
--------------------------------------------------------------- 뺨, 이마, 턱, 눈 분리를 위해서 작성한 부분
"""
#lowerb_e = np.array([0, 34, 40], dtype = "uint8")
#upperb_e = np.array([20, 255, 255], dtype = "uint8")

#lower_otsu_e = np.array([0,0,0], dtype = np.uint8)
#upper_otsu_e = np.array([0,0,0], dtype = np.uint8)

row = len(rgb_hsv_skin)

chin = int(row * 0.8)
cheekL = int(row * 0.1)
cheekL2 = int(row * 0.3)
cheek_v = int(row*0.45)
cheek_v2 = int(row*0.75)
cheekR = int(row * 0.8)
cheekR2 = int(row * 0.6)
forehead = int(row * 0.25)


chin_img = skin_otsu[chin:]
#cheekR_img = rgb_hsv_skin[cheek_v:cheek_v2, cheekR2:cheekR]
#cheekL_img = rgb_hsv_skin[cheek_v:cheek_v2, cheekL:cheekL2]

cheekR_img = rgb_skin[cheek_v:cheek_v2, cheekR2:cheekR]
cheekL_img = rgb_skin[cheek_v:cheek_v2, cheekL:cheekL2]
forehead_img = skin_otsu[:forehead]

cheek_img = np.concatenate((cheekL_img, cheekR_img), axis =1)

"""
--------------------------------------------------------------- 뺨, 이마, 턱, 눈 분리를 위해서 작성한 부분
"""

"""
haar 로 뽑아낸 눈으로 마스크 해보기---------------------------
"""
#haar_eye_size = cv.resize(haar_eye, dsize = (80, 80), interpolation=cv.INTER_LINEAR)

"""
눈동자 뽑기 ------------------------------------------------------------------------------------------
"""


"""
rowe = len(haar_eye_size)

eyehigh = int(rowe * 0.65)
eyelow = int(rowe * 0.2)

eyeleft = int(rowe * 0.2)
eyeright =int(rowe * 0.8)


haar_eye_size_crop = haar_eye_size[eyelow : eyelow+eyehigh, eyeleft: eyeleft + eyeright]
cpy = haar_eye_size[eyelow : eyelow+eyehigh, eyeleft: eyeleft + eyeright]
"""


"""

"""
# 컨투어로 눈동자 뽑기
"""
iris = contour_iris.contour_eye(cpy)

"""
# 눈동자 마스킹 + 피부를 최대한 뺴기 위해서 - otsu
"""
if iris is not None:
    
    iet, iris_otsu = cv.threshold(cv.cvtColor(iris, cv.COLOR_BGR2GRAY), -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU )
    bgr_iris_otsu = cv.cvtColor(iris_otsu, cv.COLOR_GRAY2BGR)
    
    iris_only_skin= cv.inRange(bgr_iris_otsu, lower_otsu_e, upper_otsu_e)
    iris_otsu_skin = cv.bitwise_and(iris, iris, mask = iris_only_skin)

"""

"""

"""
# 눈동자 마스킹 - Ycrcb
"""
masked_iris_ycrcb = None
yiris_otsu_skin = None
if iris is not None:
    lower_y = np.array([0,0,122], dtype = np.uint8)
    upper_y = np.array([255,173,255], dtype = np.uint8)
    iris_ycr = cv.cvtColor(iris, cv.COLOR_BGR2YCR_CB)
    iris_mask_ycrcb = cv.inRange(iris_ycr, lower_y, upper_y)
    ycrcb_masking_iris = cv.bitwise_and(iris_ycr, iris_ycr, mask = iris_mask_ycrcb)
    masked_iris_ycrcb = cv.cvtColor(ycrcb_masking_iris, cv.COLOR_YCR_CB2BGR)
    
    yet, yiris_otsu = cv.threshold(cv.cvtColor(masked_iris_ycrcb, cv.COLOR_BGR2GRAY), -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU )
    bgr_yiris_otsu = cv.cvtColor(yiris_otsu, cv.COLOR_GRAY2BGR)
    
    yiris_only_skin = cv.inRange(bgr_yiris_otsu, lower_otsu_e, upper_otsu_e)
    yiris_otsu_skin = cv.bitwise_and(masked_iris_ycrcb, masked_iris_ycrcb, mask = yiris_only_skin)

"""



""" # 그레이스케일로 변환한 이미지에서 눈동자 찾고 검출하기 (정확하지 않음..)
width, height, channel = haar_eye_size_crop.shape

inverted_eye = 255 - haar_eye_size_crop

gray_inverted = cv.cvtColor(inverted_eye, cv.COLOR_BGR2GRAY)

gray_range = cv.inRange(gray_inverted, 185, 235)

eye_extracted = cv.bitwise_and(haar_eye_size_crop, haar_eye_size_crop, mask = gray_range)
"""


"""
그레이스케일 이미지에서 가장 값이 높은 곳을 눈동자로 판단. 그곳의 좌표 검출 - 폐기
"""
"""
max_rgb_value = np.amax(gray_inverted, axis=(0, 1)) 
max_rgb_index = np.unravel_index(np.argmax(gray_inverted, axis=None), gray_inverted.shape[:2]) 
"""

"""
base = cv.rectangle(haar_eye_size_crop, (0, 0), (height, width), (0,0,0), thickness=cv.FILLED) ############# <- 눈동자 뽑기 위한 마스크
circle_img = cv.circle(base, max_rgb_index, 6, (255, 255, 255), thickness = cv.FILLED) ############## 색상반전한 이미지에서 찾아낸 눈동자 중심에서 원 그리기

#pupil = cv.bitwise_and(haar_eye_size_crop, haar_eye_size_crop, mask = circle_img)    ####### <<<<<<<<<< 남은건 circle_img로 마스크 만들어서 눈동자 뽑기
gray_circle = cv.cvtColor(circle_img, cv.COLOR_BGR2GRAY)

pupil_mask = cv.bitwise_and(gray_inverted, gray_inverted, mask = gray_circle)
"""

"""
"""
# 눈동자 뽑기 ------------------------------------------------------------------------------------------
"""

het , heye_otsu = cv.threshold(cv.cvtColor(haar_eye_size, cv.COLOR_BGR2GRAY), -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
bgr_heye_otsu = cv.cvtColor(heye_otsu, cv.COLOR_GRAY2BGR)


heye_only_skin = cv.inRange(bgr_heye_otsu, lower_otsu_e, upper_otsu_e)
heye_otsu_skin = cv.bitwise_and(haar_eye_size, haar_eye_size, mask = heye_only_skin)
"""


"""
hsv_heye = cv.cvtColor(heye_otsu_skin, cv.COLOR_BGR2HSV)
hsv_heye_mask = cv.inRange(hsv_heye, lowerb_e, upperb_e)
hsv_heye_skin_skin = cv.bitwise_and(hsv_heye, hsv_heye, mask = hsv_heye_mask)
rgb_heye_skin_skin = cv.cvtColor(hsv_heye_skin_skin, cv.COLOR_HSV2BGR)
"""

"""
--------------------------------------------------------------- 뺨, 이마, 턱, 눈 분리를 위해서 작성한 부분
"""


"""
eye_img_size = cv.resize(eye_img, dsize = (250, 100), interpolation=cv.INTER_LINEAR)

et, eye_otsu = cv.threshold(cv.cvtColor(eye_img_size, cv.COLOR_BGR2GRAY), -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
bgr_eye_otsu = cv.cvtColor(eye_otsu, cv.COLOR_GRAY2BGR)

eye_only_skin = cv.inRange(bgr_eye_otsu, lower_otsu_e, upper_otsu_e)


eye_otsu_skin = cv.bitwise_and(eye_img_size,eye_img_size, mask = eye_only_skin) ######## <<----- 눈동자 검출용 ????????????
hsv_eye_skin = cv.cvtColor(eye_otsu_skin, cv.COLOR_BGR2HSV)
hsv_eye_mask = cv.inRange(hsv_eye_skin, lowerb_e, upperb_e)

hsv_eye_skin_skin = cv.bitwise_and(hsv_eye_skin,hsv_eye_skin, mask = hsv_eye_mask)
rgb_hsv_eye_skin = cv.cvtColor(hsv_eye_skin_skin, cv.COLOR_HSV2BGR)
"""
#palette_contour = None
#palette_contour_mask = None

clt_chin = clt3.fit(chin_img.reshape(-1,3))
clt_cheek = clt4.fit(cheek_img.reshape(-1, 3))
clt_forehead = clt5.fit(forehead_img.reshape(-1,3))
#clt_eye = clt6.fit(heye_otsu_skin.reshape(-1,3))

"""
if iris is not None:
    clt_contour = clt7.fit(iris.reshape(-1,3))
    palette_contour = palette(clt_contour)
    clt_eye_mask = clt8.fit(iris_otsu_skin.reshape(-1,3))
    palette_contour_mask = palette(clt_eye_mask)
"""
palette_chin = palette(clt_chin)
palette_cheek = palette(clt_cheek)
palette_forehead = palette(clt_forehead)
#palette_eye = palette(clt_eye)


"""
--------------------------------------------------------------- 뺨, 이마, 턱, 눈 분리를 위해서 작성한 부분
"""
palette_img = palette(clt_1)

# otsu 팔레트
clt_2 = clt2.fit(rgb_hsv_skin.reshape(-1, 3))
palette_otsu = palette(clt_2)

#palette_otsu_lab = cv.cvtColor(palette_otsu, cv.COLOR_BGR2LAB)

skin_color = (0,0,0)

#lab_palette = cv.cvtColor(palette_img, cv.COLOR_BGR2LAB)

"""
팔레트 영역에서 배경과 가장 많이 나온 피부색 중, 피부색 선택하기
"""
max_area = -1
# 5개의 후보군 중 배경을 제외하고 가장 큰 면적을 가진 피부색 선택
"""
for row in palette_img:
    for color in np.unique(row, axis=0):
        if color[2] >= 70:
            current_area = np.sum(np.all(row == color, axis=-1))
            if current_area > max_area:
                max_area = current_area
                skin_color = color
"""

"""
# 5개의 후보군 중 R 값이 가장 높은 피부색 선택 - ycb 마스크
for row in palette_otsu:
    for color in row:
        if max_area < color[2]:
            max_area = color[2]
            skin_color = color

color_scalar = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
#rec = np.ones((50, 50, 3), dtype=np.uint8) * 255

#cv.rectangle(rec, (0, 0), (49, 49), color_scalar_otsu, thickness=cv.FILLED)
"""



"""
# 5개의 후보군 중 R 값이 가장 높은 피부색 선택 - otsu
for row in palette_otsu:
    for color in row:
        if max_area < color[2]:
            max_area = color[2]
            skin_color = color
"""

#color_list = []
most_common_array = []
# color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
# color_scalar_eye = (0,0,0)
color_scalar_chin = (0,0,0)
color_scalar_cheek = (0,0,0)
color_scalar_forehead = (0,0,0)
#color_scalar_contour = (0,0,0)
#color_scalar_contour_mask = (0,0,0)

"""
try:
    for row in palette_otsu:
        for color in row:
            color_list.append(color)
            
    filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0]))]
    freq = Counter(map(tuple, filtered_array_list))
            
    most_common_array, most_common_count = freq.most_common(1)[0]
    
    print("----------------- 팔레트 빈도수")
    print(f"배열 빈도수: {freq}")
    print(f"가장 빈도수가 큰 배열: {np.array(most_common_array)}")
    print(f"빈도수: {most_common_count}")
    print("----------------------------")
#color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
    color_scalar_otsu = (int(most_common_array[0]),int(most_common_array[1]) , int(most_common_array[2]))
    
except Exception as e:
    print("피부를 검출할 수 없는 이미지")
    os.remove(input_path + "/" + name + "/" + name + num + '.jpg')
"""

"""
"""
# 팔레트에 나타난 색 중 가장 큰 면적의 색상 선택하기[배경 (0,0,0) 제외]) ------------------------------------------------------------
"""
try:
    for row in palette_eye:
        for color in row:
            color_list.append(color)
            
    filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0])) and arr[2] <= 65]
    freq = Counter(map(tuple, filtered_array_list))
            
    most_common_array, most_common_count = freq.most_common(1)[0]

    print("----------------- 팔레트 빈도수")
    print(f"배열 빈도수: {freq}")
    print(f"가장 빈도수가 큰 배열: {np.array(most_common_array)}")
    print(f"빈도수: {most_common_count}")
    print("----------------------------")

#color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
    color_scalar_eye = (int(most_common_array[0]),int(most_common_array[1]) , int(most_common_array[2]))
    
except Exception as e:
    print("피부를 검출할 수 없는 이미지")
color_list.clear()

"""


try:
    color_scalar_chin = extract_most_common_color(palette_chin)
except Exception as e:
    print("피부를 검출할 수 없는 이미지")


try:
    color_scalar_cheek = extract_most_common_color(palette_cheek)
except Exception as e:
    print("피부를 검출할 수 없는 이미지")


try:
    color_scalar_forehead = extract_most_common_color(palette_forehead)
except Exception as e:
    print("피부를 검출할 수 없는 이미지")



"""
"""
# 컨투어에서 가장 면적이 큰 색상 선택하기 --------------------------------------------------------------
"""
try:
    for row in palette_contour:
        for color in row:
            color_list.append(color)
            
    filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0]))]
    freq = Counter(map(tuple, filtered_array_list))
            
    most_common_array, most_common_count = freq.most_common(1)[0]

#color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
    color_scalar_contour = (int(most_common_array[0]),int(most_common_array[1]) , int(most_common_array[2]))
    
except Exception as e:
    print("눈동자 추출을 못했어요")
color_list.clear()
"""

"""
# 컨투어에서 가장 면적이 큰 색상 선택하기 --------------------------------------------------------------
"""

"""
"""
# 마스킹한 컨투어에서 가장 면적이 큰 색상 선택하기 ---------------------------------------------------------
"""
try:
    for row in palette_contour_mask:
        for color in row:
            color_list.append(color)
            
    filtered_array_list = [arr for arr in color_list if not np.array_equal(arr, np.array([0, 0, 0]))]
    freq = Counter(map(tuple, filtered_array_list))
            
    most_common_array, most_common_count = freq.most_common(1)[0]

#color_scalar_otsu = (int(skin_color[0]), int(skin_color[1]), int(skin_color[2]))
    color_scalar_contour_mask = (int(most_common_array[0]),int(most_common_array[1]) , int(most_common_array[2]))
    
except Exception as e:
    print("눈동자 추출을 못했어요")
color_list.clear()

"""
# 마스킹한 컨투어에서 가장 면적이 큰 색상 선택하기 ---------------------------------------------------------
"""
"""

"""
팔레트에 나타난 색 중 가장 큰 면적의 색상 선택하기[배경 (0,0,0) 제외]) ------------------------------------------------------------
"""


"""
"""
# 팔레트에서 선택한 색상으로 50x50 사이즈 사각형 만들기 ---------------------------------------------------------------------------
"""
     
#otsu_rec = np.ones((50, 50, 3), dtype = np.uint8) * 255 # 피부
#otsu_eye = np.ones((50, 50, 3), dtype = np.uint8) * 255 # 눈
otsu_chin = np.ones((50,50,3), dtype = np.uint8) * 255 # 턱
otsu_cheek = np.ones((50, 50, 3), dtype = np.uint8) * 255 # 뺨
otsu_forehead = np.ones((50,50,3), dtype = np.uint8) * 255 # 이마
#otsu_contour = np.ones((50, 50, 3), dtype = np.uint8) * 255 # 컨투어
#otsu_contour_mask = np.ones((50, 50, 3), dtype = np.uint8) * 255 # 마스킹한 컨투어

#cv.rectangle(otsu_rec, (0, 0), (49, 49), color_scalar, thickness=cv.FILLED)
# cv.rectangle(otsu_eye, (0, 0), (49, 49), color_scalar_eye, thickness=cv.FILLED)

cv.rectangle(otsu_chin, (0,0), (49, 49), color_scalar_chin, thickness=cv.FILLED)
cv.rectangle(otsu_cheek, (0,0), (49, 49), color_scalar_cheek, thickness=cv.FILLED)
cv.rectangle(otsu_forehead, (0,0), (49, 49), color_scalar_forehead, thickness=cv.FILLED)


#cv.rectangle(otsu_contour, (0,0), (49, 49), color_scalar_contour, thickness=cv.FILLED)
#cv.rectangle(otsu_contour_mask, (0,0), (49, 49), color_scalar_contour_mask, thickness=cv.FILLED)

rgb_skin_color = (skin_color[2], skin_color[1], skin_color[0])
image = Image.new("RGB", (50, 50), rgb_skin_color)

"""
# 팔레트에서 선택한 색상으로 50x50 사이즈 사각형 만들기 ---------------------------------------------------------------------------
"""
"""
hsv_v = r2h.cvtcolor(skin_color)[2]
hsv_s = r2h.cvtcolor(skin_color)[1]



r_g_chin = color_scalar_chin[2] / 255
g_g_chin = color_scalar_chin[1] / 255
b_g_chin = color_scalar_chin[0] / 255


r_g_cheek = color_scalar_cheek[2] / 255
g_g_cheek = color_scalar_cheek[1] / 255
b_g_cheek = color_scalar_cheek[0] / 255

r_g_forehead = color_scalar_forehead[2] / 255
g_g_forehead = color_scalar_forehead[1] / 255
b_g_forehead = color_scalar_forehead[0] / 255

rgb_chin = sRGBColor(r_g_chin , g_g_chin, b_g_chin)
rgb_cheek = sRGBColor(r_g_cheek , g_g_cheek, b_g_cheek)
rgb_forehead = sRGBColor(r_g_forehead , g_g_forehead, b_g_cheek)


#color_b = r2l.cvtcolor(skin_color)[2]
#color_a = r2l.cvtcolor(skin_color)[1]
#color_l = r2l.cvtcolor(skin_color)[0]

color_b_chin = convert_color(rgb_chin, LabColor, through_rgb_type=sRGBColor).lab_b
color_a_chin = convert_color(rgb_chin, LabColor, through_rgb_type=sRGBColor).lab_a
color_l_chin = convert_color(rgb_chin, LabColor, through_rgb_type=sRGBColor).lab_l

color_b_cheek = convert_color(rgb_cheek, LabColor, through_rgb_type=sRGBColor).lab_b
color_a_cheek = convert_color(rgb_cheek, LabColor, through_rgb_type=sRGBColor).lab_a
color_l_cheek = convert_color(rgb_cheek, LabColor, through_rgb_type=sRGBColor).lab_l

color_b_forehead = convert_color(rgb_forehead, LabColor, through_rgb_type=sRGBColor).lab_b
color_a_forehead = convert_color(rgb_forehead, LabColor, through_rgb_type=sRGBColor).lab_a
color_l_forehead = convert_color(rgb_forehead, LabColor, through_rgb_type=sRGBColor).lab_l


hsv_h_chin = convert_color(rgb_chin, HSVColor, through_rgb_type=sRGBColor).hsv_h
hsv_v_chin = convert_color(rgb_chin, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
hsv_s_chin = convert_color(rgb_chin, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100

hsv_h_cheek = convert_color(rgb_cheek, HSVColor, through_rgb_type=sRGBColor).hsv_h
hsv_v_cheek = convert_color(rgb_cheek, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
hsv_s_cheek = convert_color(rgb_cheek, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100

hsv_h_forehead = convert_color(rgb_forehead, HSVColor, through_rgb_type=sRGBColor).hsv_h
hsv_v_forehead = convert_color(rgb_forehead, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
hsv_s_forehead = convert_color(rgb_forehead, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100

#hsv_v = r2h.cvtcolor(skin_color)[2]
#sv_s = r2h.cvtcolor(skin_color)[1]
#hsv_h = r2h.cvtcolor(skin_color)[0]

#print(color_b_cheek, color_a_cheek)

"""
# 가중치
"""
model_tone = CatBoostClassifier()
model_tone.load_model('./model/model_tone.cbm')

model_weather = CatBoostClassifier()
model_weather.load_model('./model/model_weather.cbm')

s_data = {
    'b': [color_b_forehead],
    'b_chin': [color_b_chin],
    'b_cheek': [color_b_cheek],
    's': [hsv_s_forehead],
    's_chin': [hsv_s_chin],
    's_cheek': [hsv_s_cheek],
    'h': [hsv_h_forehead],
    'h_chin': [hsv_h_chin],
    'h_cheek': [hsv_h_cheek]
}

m_data = {
    'a': [color_a_forehead],
    'a_chin': [color_a_chin],
    'a_cheek': [color_a_cheek],
    'l': [color_l_forehead],
    'l_chin': [color_l_chin],
    'l_cheek': [color_l_cheek],
    's': [hsv_s_forehead],
    's_chin': [hsv_s_chin],
    's_cheek': [hsv_s_cheek],
    'v': [hsv_v_forehead],
    'v_chin': [hsv_v_chin],
    'v_cheek': [hsv_v_cheek]
}

cat_mood = ''

df = pd.DataFrame(s_data)
df2 = pd.DataFrame(m_data)

pool = Pool(df)
pool2 = Pool(df2)

c1 = model_tone.predict(pool)
p1 = model_tone.predict_proba(pool)
print("saved model : ", c1[0])

c2 = model_weather.predict(pool)
p2 = model_weather.predict_proba(pool)
print("saved model : ", c2[0][0])

if c1[0] == 'warm':
    c1[0] = '웜'
else:
    c1[0] = '쿨'


if c2[0][0] == 'spring':
    c2[0][0] = '봄'
    model_mood = CatBoostClassifier()
    model_mood.load_model('./model/model_spring_mood.cbm')
    cat_mood = model_mood.predict(pool2)[0]
elif c2[0][0] == 'summer':
    c2[0][0] = '여름'
    model_mood = CatBoostClassifier()
    model_mood.load_model('./model/model_summer_mood.cbm')
    cat_mood = model_mood.predict(pool2)[0]
elif c2[0][0] == 'fall':
    c2[0][0] = '가을'
    model_mood = CatBoostClassifier()
    model_mood.load_model('./model/model_fall_mood.cbm')
    cat_mood = model_mood.predict(pool2)[0]
else:
    c2[0][0] = '겨울'
    model_mood = CatBoostClassifier()
    model_mood.load_model('./model/model_winter_mood.cbm')
    cat_mood = model_mood.predict(pool2)[0]

print("saved model : ", cat_mood)

if cat_mood == 'bright':
    cat_mood = '브라이트'
elif cat_mood =='light':
    cat_mood = '라이트'
elif cat_mood == 'mute':
    cat_mood = '뮤트'
else:
    cat_mood = '딥'
    
result = c1[0] + " " +  c2[0][0] + " " + cat_mood
print(result)

with open('./result/result.txt', 'w') as file:
    file.write(result)

"""
f_w = color_b_forehead * 0.5
chin_w = color_b_chin * 0.7
cheek_w = color_b_cheek * 0.8
w_sum = f_w + chin_w + cheek_w

f_s_w = hsv_s_forehead * 0.4
chin_s_w = hsv_s_chin * 0.7
cheek_s_w = hsv_s_cheek * 0.8
s_sum = f_s_w + chin_s_w + cheek_s_w


"""
# 추출한 색상으로 피부톤 진단하기 - 진행 중 .. (폐기할듯)
"""
#if color_b <= 18.5:
if w_sum <= 19.0:
    tone = True

if tone:
    #print ("쿨톤")
    #if hsv_s >= 33:
    if s_sum >= 27.5:
        #print(" 겨울")
        if hsv_v <= -7:
            #print(" 브라이트")
            final_tone = tone_list[7]
        else:
            #print(" 딥")
            final_tone = tone_list[6]
    
    else:
        #print(" 여름")
        if hsv_v - v_for_summer >= 0:
            #print(" 라이트")
            final_tone = tone_list[2]
        else:
            #print(" 뮤트")
            final_tone = tone_list[3]
            
else:
    #print("웜톤")
    if s_sum <= 44.4:
        #print(" 봄")
        #if hsv_s >= 33:
        if hsv_s - s_for_warm <= 1.5:
            #print(" 브라이트")
            final_tone = tone_list[0]
        else:
            #print(" 라이트")
            final_tone = tone_list[1]
        
    else:
        #print(" 가을")
        if hsv_s - s_for_warm <= 0.4:
            #print(" 딥")
            final_tone = tone_list[5]
        else:
            #print(" 뮤트")
            final_tone = tone_list[4]
            
            
print(final_tone)
#print(cat_tone, cat_weather)
"""
end_time = time.time()
elapsed_time = end_time - start_time


"""
하한 - 상한 사이의 값들은 백색, 나머지는 흑색. - 사용안함
"""
#thresh_img1 = cv.inRange(gray_img, threshold, 255)

"""
"""
# 이미지 확인하기 -----------------------------------------------------------------------------------------------------------
"""

combined_img = np.concatenate((crop_img, skin_otsu,rgb_hsv_skin), axis =1)
combined_palette = np.concatenate((palette_chin, palette_cheek, palette_forehead), axis = 0)
combined_color = np.concatenate((otsu_chin, otsu_cheek, otsu_forehead), axis = 0)
#white_origin = np.concatenate((origin_img, img), axis = 1)
#inverted_combine = np.concatenate((gray_inverted, gray_range), axis = 0)

#eye = np.concatenate((eye_otsu_skin, rgb_hsv_eye_skin), axis = 0)

#combined_skin = np.concatenate((skin, rgb_skin, lab_skin), axis =1)
#cv.imshow('combined skin', combined_skin)

cv.imshow('combined', combined_img) ######################

#cv.imshow('palette', palette_img)
#cv.imshow('lab palette', lab_palette)
#cv.imshow('lab skin', lab_rec)
"""
# cv.imshow('otsu palette', palette_otsu) ######################
"""
#cv.imshow('otsu palette lab', palette_otsu_lab)


#cv.imshow('otsu color', otsu_rec)
#cv.imshow('otsu eye color', otsu_eye)


#cv.imshow('otsu lab', otsu_lab)


#v.imshow('chin', chin_img)
#cv.imshow('cheek', cheek_img)
#cv.imshow('forehead', forehead_img)
#cv.imshow('eye', eye_img_size)

cv.imshow('chin - cheek - forehead', combined_palette)
cv.imshow('chin', chin_img)
cv.imshow('cheek', cheek_img)
cv.imshow('forehead', forehead_img)
cv.imshow('color', combined_color)

#cv.imshow('eye masking', eye)
#cv.imshow('bgr eye otsu', bgr_eye_otsu)
#cv.imshow('eye only skin', eye_only_skin)
#cv.imshow('eye size', haar_eye_size)
#cv.imshow('heye only skin', heye_only_skin)
#cv.imshow('heye otsu skin', heye_otsu_skin)
#cv.imshow('eye crop 2', haar_eye_size_crop)
#cv.imshow('cheek color', otsu_cheek)


#cv.imshow('ycbcr chin', chin_img)

#cv.imshow('gray & binary', inverted_combine)

#cv.imshow('cirlcle', gray_circle)
#cv.imshow('pupil gray', pupil_mask)
"""

"""
cv.imshow('eye origin', cpy)
if iris is not None:
    iris_combined = np.concatenate((iris, iris_otsu_skin), axis = 0)
    iris_color_combined = np.concatenate((otsu_contour, otsu_contour_mask), axis = 0)
    iris_palette_combined = np.concatenate((palette_contour, palette_contour_mask), axis = 0)
    #cv.imshow('iris', iris)
    #cv.imshow('iris color' , otsu_contour)
    #cv.imshow('iris palette', palette_contour)
    #cv.imshow('iris_otsu', iris_otsu_skin)
    #cv.imshow('iris_otsu_mask', otsu_contour_mask)
    #cv.imshow('iris otsu palette', palette_contour_mask)
    cv.imshow('iris', iris_combined)
    cv.imshow('iris color', iris_color_combined)
    cv.imshow('iris palette', iris_palette_combined)
    cv.imshow('ycrcb' , masked_iris_ycrcb)
    cv.imshow('ycrcb + otsu', yiris_otsu_skin)
"""

"""
#cv.imshow('pupil extracted', pupil_mask_bgr)
#cv.imshow('rgb skin skin', rgb_heye_skin_skin)

cv.waitKey(0)

cv.destroyAllWindows()

print(f"경과 시간: {elapsed_time} 초")

"""
#이미지 확인하기 -----------------------------------------------------------------------------------------------------------
"""

"""