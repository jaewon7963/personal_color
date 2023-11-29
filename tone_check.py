import cv2 as cv
import os
import math
import numpy as np
import glob
import pandas as pd
import cat
from colormath.color_objects import XYZColor, sRGBColor, LabColor, HSVColor
from colormath.color_conversions import convert_color
from catboost import CatBoostClassifier, Pool

import rgb2hsv as r2h
import rgb2xyz2lab as r2l

model_tone = CatBoostClassifier()
model_tone.load_model('./model/model_tone2.cbm')

#model_weather = CatBoostClassifier()
#model_weather.load_model('./model/model_weather2.cbm')

gt_list = ['spring_warm_bright' , 'spring_warm_light', 'autumn_warm_mute', 'autumn_warm_deep',
           'summer_cool_mute', 'summer_cool_light', 'winter_cool_bright', 'winter_cool_deep']


# 모델 정확도를 측정하기 위해 gt 값을 설정합니다.
gt = gt_list[6]


# 테스트에 사용할 입력들(어떤 톤인지 결정된 이미지들)을 결정합니다.

name = 'winter_result_w'
input_path = './result/forehead/' + name
tone_list = ["봄 웜 브라이트" , "봄 웜 라이트", "여름 쿨 라이트", "여름 쿨 뮤트", "가을 웜 뮤트", "가을 웜 딥", "겨울 쿨 딥", "겨울 쿨 브라이트"]

goal = ''

tone_c = ''
weather =''
mood = ''

"""
thresh = 4.0
s_for_cool = 25.3
v_for_warm = 88
s_for_warm = 20.3
v_for_summer = 88.5
h_for_cool = 17.7
"""

if 'cool' in gt:
    tone_c = '쿨'
else:
    tone_c = '웜'
    
if 'bright' in gt:
    mood = '브라이트'
elif 'light' in gt:
    mood = '라이트'
elif 'mute' in gt:
    mood = '뮤트'
elif 'deep' in gt:
    mood = '딥'
    
if 'spring' in gt:
    weather = '봄'
elif 'summer' in gt:
    weather = '여름'
elif 'autumn' in gt:
    weather = '가을'
elif 'winter' in gt:
    weather = '겨울'

if gt == 'spring_warm_bright':
    goal = '봄 웜 브라이트'
elif gt == 'spring_warm_light':
    goal = '봄 웜 라이트'
elif gt == 'summer_cool_light':
    goal = '여름 쿨 라이트'
elif gt == 'summer_cool_mute':
    goal = '여름 쿨 뮤트'
elif gt == 'autumn_warm_mute':
    goal = '가을 웜 딥'
elif gt == 'autumn_warm_deep':
    goal = '가을 웜 뮤트'
elif gt == 'winer_cool_deep':
    goal = '겨울 쿨 딥'
else:
    goal = '겨울 쿨 브라이트'    

tone_correct = 0
weather_correct = 0
mood_correct = 0

cat_tone_correct = 0
cat_weather_correct = 0
cat_mood_correct = 0


num = 0
count = 0
b_sum = 0

b_list =[]
b_list_chin = []
b_list_cheek = []

a_list =[]
a_list_chin = []
a_list_cheek = []

l_list =[]
l_list_chin = []
l_list_cheek = []

v_list =[]
v_list_chin = []
v_list_cheek = []

s_list =[]
s_list_chin = []
s_list_cheek = []

h_list = []
h_list_chin = []
h_list_cheek = []


difv_for_summer =[]
dif_h = []

print(tone_c)
print(weather)
print(mood)

for file in os.listdir(input_path):
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
        final_tone = ""
        tone = False
        print(file)
        #name = 'warm_s'
        
        """
        if file.endswith('_01.png'):
            name = file.split('_01.png')[0] + '.png'
        elif file.endswith('_02.png'):
            name = file.split('_02.png')[0] + '.png'
        """
        
        
        cheek_dir = './result/cheek/' + name
        chin_dir = './result/chin/' + name
        

        if not os.path.exists(chin_dir + "/" + file):
            try:
                os.remove(input_path + "/" + file)
                os.remove(cheek_dir + "/" + file)
            except Exception as chin:
                continue
            continue
        
        if not os.path.exists(cheek_dir + "/" + file):
            try:
                os.remove(input_path + "/" + file)
                os.remove(chin_dir + "/" + file)
            except Exception as cheek:
                continue
            continue
        
        
        img = cv.imread(input_path + "/" + file)
        img_chin = None
        img_cheek = None
        
        img_chin = cv.imread(chin_dir + "/" + file)
        img_cheek = cv.imread(cheek_dir + "/" + file)
        
        
        """
        cheek = cv.imread('./result/cool_cheek/' + cheek_name)
        if cheek is None:
            continue
        """
        
        r_g = img[20,20][2] / 255
        g_g = img[20,20][1] / 255
        b_g = img[20,20][0] / 255
        
        #"""
        r_g_chin = img_chin[20,20][2] / 255
        g_g_chin = img_chin[20,20][1] / 255
        b_g_chin = img_chin[20,20][0] / 255
        
        r_g_cheek = img_cheek[20,20][2] / 255
        g_g_cheek = img_cheek[20,20][1] / 255
        b_g_cheek = img_cheek[20,20][0] / 255
        #"""
        
        
        #r_g_c = cheek[20,20][2] / 255
        #g_g_c = cheek[20,20][1] / 255
        #b_g_c = cheek[20,20][0] / 255
        
        rgb = sRGBColor(r_g , g_g, b_g)
        #"""
        rgb_chin = sRGBColor(r_g_chin , g_g_chin, b_g_chin)
        rgb_cheek = sRGBColor(r_g_cheek , g_g_cheek, b_g_cheek)
        #"""
        #rgb_c = sRGBColor(r_g_c, g_g_c, b_g_c)
        
        
        #"""
        color_b = r2l.cvtcolor(img[20,20])[2]
        color_a = r2l.cvtcolor(img[20,20])[1]
        color_l = r2l.cvtcolor(img[20,20])[0]
        
        hsv_h = r2h.cvtcolor(img[20,20])[0]
        hsv_v = r2h.cvtcolor(img[20,20])[2]
        hsv_s = r2h.cvtcolor(img[20,20])[1]
        #"""
        
        
        color_b = convert_color(rgb, LabColor, through_rgb_type=sRGBColor).lab_b
        color_a = convert_color(rgb, LabColor, through_rgb_type=sRGBColor).lab_a
        color_l = convert_color(rgb, LabColor, through_rgb_type=sRGBColor).lab_l
        
        #"""
        color_b_chin = convert_color(rgb_chin, LabColor, through_rgb_type=sRGBColor).lab_b
        color_a_chin = convert_color(rgb_chin, LabColor, through_rgb_type=sRGBColor).lab_a
        color_l_chin = convert_color(rgb_chin, LabColor, through_rgb_type=sRGBColor).lab_l
        
        color_b_cheek = convert_color(rgb_cheek, LabColor, through_rgb_type=sRGBColor).lab_b
        color_a_cheek = convert_color(rgb_cheek, LabColor, through_rgb_type=sRGBColor).lab_a
        color_l_cheek = convert_color(rgb_cheek, LabColor, through_rgb_type=sRGBColor).lab_l
        #"""
        
        hsv_h = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor).hsv_h
        hsv_v = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
        hsv_s = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100
        
        #"""
        hsv_h_chin = convert_color(rgb_chin, HSVColor, through_rgb_type=sRGBColor).hsv_h
        hsv_v_chin = convert_color(rgb_chin, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
        hsv_s_chin = convert_color(rgb_chin, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100
        
        hsv_h_cheek = convert_color(rgb_cheek, HSVColor, through_rgb_type=sRGBColor).hsv_h
        hsv_v_cheek = convert_color(rgb_cheek, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
        hsv_s_cheek = convert_color(rgb_cheek, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100
        #"""
        
        """
        color_b_c = convert_color(rgb_c, LabColor, through_rgb_type=sRGBColor).lab_b
        color_a_c = convert_color(rgb_c, LabColor, through_rgb_type=sRGBColor).lab_a
        color_l_c = convert_color(rgb_c, LabColor, through_rgb_type=sRGBColor).lab_l
        
        
        hsv_h_c = convert_color(rgb_c, HSVColor, through_rgb_type=sRGBColor).hsv_h
        hsv_v_c = convert_color(rgb_c, HSVColor, through_rgb_type=sRGBColor).hsv_v * 100
        hsv_s_c = convert_color(rgb_c, HSVColor, through_rgb_type=sRGBColor).hsv_s * 100
        """
        #lab = convert_color(rgb, LabColor, through_rgb_type=sRGBColor)
        
        f_w = color_b * 0.5
        chin_w = color_b_chin * 0.7
        cheek_w = color_b_cheek * 0.8
        w_sum = f_w + chin_w + cheek_w
        
        f_s_w = hsv_s * 0.4
        chin_s_w = hsv_s_chin * 0.7
        cheek_s_w = hsv_s_cheek * 0.8
        s_sum = f_s_w + chin_s_w + cheek_s_w
        
        #if color_b <= 18.5:
        #if abs(color_b - color_a) >= thresh:
            #tone = True
            
        if w_sum <= 19.0:
            tone = True
            
        #if color_b <= color_a:
            #tone = True


        if tone:
            #print ("쿨톤")
            #if hsv_s >= 33:
            if s_sum >= 27.5:
                #print(" 겨울")
                if hsv_v  <= -7:
                    #print(" 브라이트")
                    final_tone = tone_list[7]
                else:
                    #print(" 딥")
                    final_tone = tone_list[6]
            
            else:
                #print(" 여름")
                if hsv_v >= 0:
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
                if hsv_s  <= 1.5:
                    #print(" 브라이트")
                    final_tone = tone_list[0]
                else:
                    #print(" 라이트")
                    final_tone = tone_list[1]
                
            else:
                #print(" 가을")
                if hsv_s  <= 0.4:
                    #print(" 딥")
                    final_tone = tone_list[5]
                else:
                    #print(" 뮤트")
                    final_tone = tone_list[4]
            
        #print(file + " : " + final_tone , hsv_v, hsv_s)
        if final_tone == goal:
            count += 1
            
        if tone_c in final_tone:
            tone_correct += 1
            
        if weather in final_tone:
            weather_correct += 1
            
        if mood in final_tone:
            mood_correct += 1
            
        num += 1
        b_list.append(round(color_b,2))
        #"""
        b_list_chin.append(round(color_b_chin,2))
        b_list_cheek.append(round(color_b_cheek,2))
        #"""
        
        a_list.append(round(color_a,2))
        #"""
        a_list_chin.append(round(color_a_chin,2))
        a_list_cheek.append(round(color_a_cheek,2))
        #"""
        
        l_list.append(round(color_l,2))
        #"""
        l_list_chin.append(round(color_l_chin,2))
        l_list_cheek.append(round(color_l_cheek,2))
        #"""
        
        h_list.append(round(hsv_h,2))
        #"""
        h_list_chin.append(round(hsv_h_chin,2))
        h_list_cheek.append(round(hsv_h_cheek,2))
        #"""
        
        v_list.append(round(hsv_v,2))
        #"""
        v_list_chin.append(round(hsv_v_chin,2))
        v_list_cheek.append(round(hsv_v_cheek,2))
        #"""
        
        s_list.append(round(hsv_s,2))
        #"""
        s_list_chin.append(round(hsv_s_chin,2))
        s_list_cheek.append(round(hsv_s_cheek,2))
        #"""
        cat_tone = ''
        cat_weather = ''
        cat_mood = ''
        
        s_data = {
            'b': [color_b],
            'b_chin': [color_b_chin],
            'b_cheek': [color_b_cheek],
            's': [hsv_s],
            's_chin': [hsv_s_chin],
            's_cheek': [hsv_s_cheek],
            'h': [hsv_h],
            'h_chin': [hsv_h_chin],
            'h_cheek': [hsv_h_cheek]
        }
        
        m_data = {
            'a': [color_a],
            'a_chin': [color_a_chin],
            'a_cheek': [color_a_cheek],
            'l': [color_l],
            'l_chin': [color_l_chin],
            'l_cheek': [color_l_cheek],
            's': [hsv_s],
            's_chin': [hsv_s_chin],
            's_cheek': [hsv_s_cheek],
            'v': [hsv_v],
            'v_chin': [hsv_v_chin],
            'v_cheek': [hsv_v_cheek]
        }

        df = pd.DataFrame(s_data)
        pool = Pool(df)
        
        df2 = pd.DataFrame(m_data)
        pool2 = Pool(df2)
        
        c1 = model_tone.predict(pool)
        #c2 = model_weather.predict(pool)
        
        cat_tone = c1[0]
        #cat_weather = c2[0][0]
        """
        if cat_weather == 'spring':
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_spring_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        elif cat_weather == 'summer':
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_summer_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        elif cat_weather == 'fall':
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_fall_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        else:
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_winter_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        """
        
            
        ct = ''
        cw = ''
        cm = ''
        
        if cat_tone == 'warm':
            ct = '웜'
            model_weather = CatBoostClassifier()
            model_weather.load_model('./model/model_warm.cbm')
            cat_weather = model_weather.predict(pool)
            
        elif cat_tone == 'cool':
            ct = '쿨'
            model_weather = CatBoostClassifier()
            model_weather.load_model('./model/model_cool.cbm')
            cat_weather = model_weather.predict(pool)
        
        if cat_weather == 'spring':
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_spring_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        elif cat_weather == 'summer':
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_summer_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        elif cat_weather == 'fall':
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_fall_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        else:
            model_mood = CatBoostClassifier()
            model_mood.load_model('./model/model_winter_mood.cbm')
            cat_mood = model_mood.predict(pool2)
        
        if cat_weather == 'spring':
            cw = '봄'
        elif cat_weather == 'summer':
            cw = '여름'
        elif cat_weather == 'fall':
            cw = '가을'
        elif cat_weather == 'winter':
            cw = '겨울'
            
        if cat_mood == 'bright':
            cm = '브라이트'
        elif cat_mood == 'light':
            cm = '라이트'
        elif cat_mood == 'deep':
            cm = '딥'
        elif cat_mood == 'mute':
            cm = '뮤트'
            
        if ct in goal:
            cat_tone_correct += 1
            
        if cw in goal:
            cat_weather_correct += 1
            
        if cm in goal:
            cat_mood_correct += 1
            
            
        # gt 값과 일치한 이미지들로 모델을 학습시키기 위해 일치하지 않은 이미지들은 삭제합니다.
        """
        if ct not in goal:
            try:
                os.remove(input_path + "/" + file)
                os.remove(cheek_dir + "/" + file)
                os.remove(chin_dir + "/" + file)
            except Exception as chin:
                continue
            continue
        
        if cw not in goal:
            try:
                os.remove(input_path + "/" + file)
                os.remove(cheek_dir + "/" + file)
                os.remove(chin_dir + "/" + file)
            except Exception as chin:
                continue
            continue
        """
            
        """
        if cat_tone != tone:
            os.remove(input_path + "/" + file)
            os.remove(cheek_dir + "/" + file)
            os.remove(chin_dir + "/" + file)
            continue
            
        if cat_weather != weather:
            os.remove(input_path + "/" + file)
            os.remove(cheek_dir + "/" + file)
            os.remove(chin_dir + "/" + file)
            continue
        """
        """
        if cw not in goal:
            os.remove(input_path + "/" + file)
            os.remove(cheek_dir + "/" + file)
            os.remove(chin_dir + "/" + file)
            continue
        """
        
#file_count = len(glob.glob(input_path + '/*'))
file_count = len(b_list_chin)
print("-----------------------------")
print(f"파일 개수 = {file_count}")
print("-----------------------------")
print("b값 평균(눈) = " , np.mean(b_list))
print("b값 평균(턱) = " , np.mean(b_list_chin))
print("b값 평균(뺨) = " , np.mean(b_list_cheek))
print("b값 분산 = " , np.var(b_list))
print("b값 표준편차 = ", math.sqrt(np.mean(b_list)))
print("-----------------------------")

print("a값 평균(눈) = " , np.mean(a_list))
print("a값 평균(턱)= " , np.mean(a_list_chin))
print("a값 평균(뺨) = " , np.mean(a_list_cheek))
print("a값 분산 = " , np.var(a_list))
print("a값 표준편차 = ", math.sqrt(np.mean(a_list)))

print("-----------------------------")
print("l값 평균 = " , np.mean(l_list))
print("L값 평균(턱)= " , np.mean(l_list_chin))
print("L값 평균(뺨) = " , np.mean(l_list_cheek))
print("l값 분산 = " , np.var(l_list))
print("l값 표준편차 = ", math.sqrt(np.mean(l_list)))
print("-----------------------------")

print("h값 평균 = ", np.mean(h_list))
print("h값 평균(턱)= " , np.mean(h_list_chin))
print("h값 평균(뺨) = " , np.mean(h_list_cheek))

print("-----------------------------")
print("v값 평균 = " , np.mean(v_list))
print("v값 평균(턱)= " , np.mean(v_list_chin))
print("v값 평균(뺨) = " , np.mean(v_list_cheek))
print("v값 분산 = " , np.var(v_list))
print("v값 표준편차 = ", math.sqrt(np.mean(v_list)))
print("-----------------------------")

print("-----------------------------")
print("s값 평균 = " , np.mean(s_list))
print("s값 평균(턱)= " , np.mean(s_list_chin))
print("s값 평균(뺨) = " , np.mean(s_list_cheek))
print("s값 분산 = " , np.var(s_list))
print("s값 표준편차 = ", math.sqrt(np.mean(s_list)))
print("-----------------------------")

print("V -v for summmer = ", np.mean(difv_for_summer))
print("H - h for cool = ", np.mean(dif_h))

print("-----------------------------")

print("실제 톤 = ", tone_c)
print('일치한 톤 = ', tone_correct)
print((tone_correct / file_count ) * 100)
print('cat 일치한 톤 = ', cat_tone_correct)
print((cat_tone_correct / file_count ) * 100)
print("-----------------------------")
print("실제 계절 = ", weather)
print('일치한 계절 = ', weather_correct)
print((weather_correct / file_count ) * 100)
print('cat 일치한 계절 = ', cat_weather_correct)
print((cat_weather_correct / file_count ) * 100)
print("-----------------------------")
print("실제 분위기 = ", mood)
print('일치한 분위기 = ', mood_correct)
print((mood_correct / file_count ) * 100)
print('cat 일치한 분위기 = ', cat_mood_correct)
print((cat_mood_correct / file_count ) * 100)
print("-----------------------------")
print(" 일치한 개수 = ", count)
print((count / file_count ) * 100)

print("-----------------------------")


# 선별된 데이터들로 학습을 시키기 위해 csv 파일을 작성합니다.
"""
data = {'tone': ['1'] * file_count,'weather': ['winter'] * file_count ,\
        'b': b_list, 'b_chin': b_list_chin, 'b_cheek' : b_list_cheek, \
        's': s_list, 's_chin': s_list_chin, 's_cheek' : s_list_cheek, \
        'h': h_list, 'h_chin': h_list_chin, 'h_cheek' : h_list_cheek}
"""

"""
data = {'mood': ['deep'] * file_count,\
        'b': b_list, 'b_chin': b_list_chin, 'b_cheek' : b_list_cheek, \
        'a': a_list, 'a_chin': a_list_chin, 'a_cheek' : a_list_cheek, \
        'l': l_list, 'l_chin': l_list_chin, 'l_cheek' : l_list_cheek, \
        's': s_list, 's_chin': s_list_chin, 's_cheek' : s_list_cheek, \
        'v': v_list, 'v_chin': v_list_chin, 'v_cheek' : v_list_cheek, \
        'h': h_list, 'h_chin': h_list_chin, 'h_cheek' : h_list_cheek}

#data = {'tone': ['0'] * file_count, 'b' : b_list, 'a': a_list}
df = pd.DataFrame(data)

df.to_csv('winter_deep.csv', index=False)

"""