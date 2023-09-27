import cv2 as cv
import numpy as np
import os

name = "chaeyeong2"
input_path = './image/' + name + '/'

output_dir_forehead = './result/forehead/' + name
output_dir_cheek = './result/cheek/' + name
output_dir_chin = './result/chin/' + name
output_dir = './result/eye/' + name 


if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

if not os.path.isdir(output_dir_cheek):
    os.makedirs(output_dir_cheek)
    
if not os.path.isdir(output_dir_chin):
    os.makedirs(output_dir_chin)
    
if not os.path.isdir(output_dir_forehead):
    os.makedirs(output_dir_forehead)
    
    
"""
논문에서 제안하는 threshold
  Y       Cr      Cb
0-255, 133-173, 80-120
"""
lower = np.array([0,123,75], dtype = np.uint8)
upper = np.array([255,173,135], dtype = np.uint8)

lower_otsu = np.array([255,255,255], dtype = np.uint8)
upper_otsu = np.array([255,255,255], dtype = np.uint8)

face_cascade = cv.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

"""
초기 값은
[0, 48, 80]
[20, 255, 255] 로 진행하였음
"""

lowerb = np.array([0, 24, 80], dtype = "uint8")
upperb = np.array([20, 200, 255], dtype = "uint8")

face_cascade = cv.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
eye_detector = cv.CascadeClassifier('./xml/haarcascade_eye.xml')



def eye_extracting():
    for file in os.listdir(input_path):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            count = 0
            """
            haar로 얼굴 검출하기
            """
            
            img = cv.imread(input_path +  file)
            #img = white.white_balancing(origin_img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
            for (x,y,w,h) in faces:
                cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                count += 1
                
            if count != 1:
                print("얼굴이 한개가 아니예요")
                os.remove(input_path + file)
                continue
    
            
            """
            haar로 눈 검출하기
            """
            max_len = 0
            haar_eye = None
            
            eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=2.0, minNeighbors=7)
            for (ex,ey,ew,eh) in eyes:
                if max_len < ey + eh:
                    max_len = ey + eh
                    haar_eye = roi_color[ey:ey + eh, ex:ex + ew]
                #cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
            if haar_eye is None:
                print("눈이 검출되지 않았어요.")
                continue
                
            row = len(haar_eye)
            
            eyehigh = int(row * 0.65)
            eyelow = int(row * 0.2)
            
            eyeleft = int(row * 0.2)
            eyeright =int(row * 0.8)
            
            eye_crop = haar_eye[eyelow : eyelow+eyehigh, eyeleft: eyeleft + eyeright]
    
            output_path = os.path.join(output_dir, file.split('.')[0] + '.png')
            output_path_crop = os.path.join(output_dir, file.split('.')[0] + "_cropped" + ".png")
            print(output_path)
            
            cv.imwrite(output_path_crop, haar_eye)
            #cv.imwrite(output_path, haar_eye)

def cheek_extracting():
    for file in os.listdir(input_path):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            
            count = 0
            
            img = cv.imread(input_path +  file)
            #img = white.white_balancing(origin_img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
            for (x,y,w,h) in faces:
                cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                count += 1
                
            if count != 1:
                print("얼굴이 한개가 아니예요")
                continue

                
            crop_img = img[y:y+h, x:x+w]
            crop_gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
            ycbcr_img = cv.cvtColor(crop_img, cv.COLOR_BGR2YCR_CB)
    
            t, t_otsu = cv.threshold(crop_gray, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            cvt_otsu = cv.cvtColor(t_otsu, cv.COLOR_GRAY2BGR)
            
            """
            # Ycbcr 을 이용한 마스크
            """
            ycbcr_mask = cv.inRange(ycbcr_img, lower, upper)
            ycbcr_skin = cv.bitwise_and(ycbcr_img, ycbcr_img, mask = ycbcr_mask)
            yc2bgr_skin = cv.cvtColor(ycbcr_skin, cv.COLOR_YCR_CB2BGR)
            
            
            
            only_skin = cv.inRange(cvt_otsu, lower_otsu, upper_otsu)
            
            
            skin_otsu = cv.bitwise_and(crop_img, crop_img, mask = only_skin)
            
            hsv_skin = cv.cvtColor(skin_otsu, cv.COLOR_BGR2HSV)
            hsv_mask = cv.inRange(hsv_skin, lowerb, upperb)

            hsv_skin_skin = cv.bitwise_and(hsv_skin, hsv_skin, mask = hsv_mask)
            rgb_hsv_skin = cv.cvtColor(hsv_skin_skin, cv.COLOR_HSV2BGR)
            
            row = len(rgb_hsv_skin)
            
            cheekL = int(row * 0.1)
            cheekL2 = int(row * 0.3)
            cheek_v = int(row*0.45)
            cheek_v2 = int(row*0.75)
            cheekR = int(row * 0.8)
            cheekR2 = int(row * 0.6)
            
            #cheekR_img = rgb_hsv_skin[cheek_v:cheek_v2, cheekR2:cheekR]
            #cheekL_img = rgb_hsv_skin[cheek_v:cheek_v2, cheekL:cheekL2]
            
            #cheekR_img = crop_img[cheek_v:cheek_v2, cheekR2:cheekR]
            #cheekL_img = crop_img[cheek_v:cheek_v2, cheekL:cheekL2]
            
            cheekR_img = yc2bgr_skin[cheek_v:cheek_v2, cheekR2:cheekR]
            cheekL_img = yc2bgr_skin[cheek_v:cheek_v2, cheekL:cheekL2]
            
            cheek_img = np.concatenate((cheekL_img, cheekR_img), axis =1)
    
            output_path = os.path.join(output_dir_cheek, file.split('.')[0] + '.png')
            print(output_path)
            cv.imwrite(output_path, cheek_img)
            
def chin_extracting():
    for file in os.listdir(input_path):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            
            count = 0
            
            img = cv.imread(input_path +  file)
            #img = white.white_balancing(origin_img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
            for (x,y,w,h) in faces:
                cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                count += 1
                
            if count != 1:
                print("얼굴이 한개가 아니예요")
                continue

                
            crop_img = img[y:y+h, x:x+w]
            crop_gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
            ycbcr_img = cv.cvtColor(crop_img, cv.COLOR_BGR2YCR_CB)
    
            t, t_otsu = cv.threshold(crop_gray, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            cvt_otsu = cv.cvtColor(t_otsu, cv.COLOR_GRAY2BGR)
            
            """
            # Ycbcr 을 이용한 마스크
            """
            ycbcr_mask = cv.inRange(ycbcr_img, lower, upper)
            ycbcr_skin = cv.bitwise_and(ycbcr_img, ycbcr_img, mask = ycbcr_mask)
            yc2bgr_skin = cv.cvtColor(ycbcr_skin, cv.COLOR_YCR_CB2BGR)
            
            
            
            only_skin = cv.inRange(cvt_otsu, lower_otsu, upper_otsu)
            
            
            skin_otsu = cv.bitwise_and(crop_img, crop_img, mask = only_skin)
            
            hsv_skin = cv.cvtColor(skin_otsu, cv.COLOR_BGR2HSV)
            hsv_mask = cv.inRange(hsv_skin, lowerb, upperb)

            hsv_skin_skin = cv.bitwise_and(hsv_skin, hsv_skin, mask = hsv_mask)
            rgb_hsv_skin = cv.cvtColor(hsv_skin_skin, cv.COLOR_HSV2BGR)
            
            row = len(rgb_hsv_skin)
            
            chin = int(row * 0.8)
            
            #chin_img = rgb_hsv_skin[chin:]
            #chin_img = crop_img[chin:]
            chin_img = yc2bgr_skin[chin:]
    
            output_path = os.path.join(output_dir_chin, file.split('.')[0] + '.png')
            print(output_path)
            
            cv.imwrite(output_path, chin_img)
            
def forehead_extracting():
    for file in os.listdir(input_path):
        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
            
            count = 0
            
            img = cv.imread(input_path +  file)
            #img = white.white_balancing(origin_img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
            for (x,y,w,h) in faces:
                cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                count += 1
                
            if count != 1:
                print("얼굴이 한개가 아니예요")
                continue

                
            crop_img = img[y:y+h, x:x+w]
            crop_gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
            ycbcr_img = cv.cvtColor(crop_img, cv.COLOR_BGR2YCR_CB)
    
            t, t_otsu = cv.threshold(crop_gray, -1, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            cvt_otsu = cv.cvtColor(t_otsu, cv.COLOR_GRAY2BGR)
            
            """
            # Ycbcr 을 이용한 마스크
            """
            ycbcr_mask = cv.inRange(ycbcr_img, lower, upper)
            ycbcr_skin = cv.bitwise_and(ycbcr_img, ycbcr_img, mask = ycbcr_mask)
            yc2bgr_skin = cv.cvtColor(ycbcr_skin, cv.COLOR_YCR_CB2BGR)
            
            
            
            only_skin = cv.inRange(cvt_otsu, lower_otsu, upper_otsu)
            
            
            skin_otsu = cv.bitwise_and(crop_img, crop_img, mask = only_skin)
            
            hsv_skin = cv.cvtColor(skin_otsu, cv.COLOR_BGR2HSV)
            hsv_mask = cv.inRange(hsv_skin, lowerb, upperb)

            hsv_skin_skin = cv.bitwise_and(hsv_skin, hsv_skin, mask = hsv_mask)
            rgb_hsv_skin = cv.cvtColor(hsv_skin_skin, cv.COLOR_HSV2BGR)
            
            row = len(rgb_hsv_skin)
            
            forehead = int(row * 0.25)
            
            #forehead_img = rgb_hsv_skin[:forehead]
            #forehead_img = crop_img[:forehead]
            #forehead_img = yc2bgr_skin[:forehead]
            forehead_img = skin_otsu[:forehead]

            output_path = os.path.join(output_dir_forehead, file.split('.')[0] + '.png')
            print(output_path)
            
            cv.imwrite(output_path, forehead_img)
            
#eye_extracting()
cheek_extracting()
chin_extracting()
forehead_extracting()