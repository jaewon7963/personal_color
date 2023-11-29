import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

#train = pd.read_csv('./whole.csv')
train = pd.read_csv('./weather_csv/warm.csv')
#train_tone = train['tone']
train_weather = train['weather']

#train_mood = train['weather']

x = train.drop(['weather', 'tone'], axis = 1)

#test_tone = catboost_pool = Pool(x, train_tone)
test_weather = catboost_pool = Pool(x, train_weather)

"""
model_tone = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
"""
model_weather = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)


"""
model_mood = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
"""


#model_tone.fit(x, train_tone)
model_weather.fit(x, train_weather)
#model_mood.fit(x, train_mood)

#model_tone.save_model('model_tone.cbm')
model_weather.save_model('./model/model_warm.cbm')
#model_mood.save_model('model_winter_mood.cbm')

"""
preds_class = model_tone.predict(x)
preds_proba = model_tone.predict_proba(test_tone)
#print("class = ", preds_class)
#print("proba = ", preds_proba)


preds_class = model_weather.predict(test_weather)
preds_proba = model_weather.predict_proba(test_weather)
#print("class = ", preds_class)
#print("proba = ", preds_proba)
"""

"""
s_data = {
    'b': [26.78476],
    'b_chin': [12.355],
    'b_cheek': [15.288],
    's': [30.12048],
    's_chin': [22.5663],
    's_cheek': [27.510917],
    'h': [295.2],
    'h_chin': [17.647],
    'h_cheek': [17.1428]
}
df = pd.DataFrame(s_data)

pool = Pool(df)
c1 = model_tone.predict(pool)
p1 = model_tone.predict_proba(pool)
print('c1 = ', c1)
print('p1 = ', p1)

c2 = model_weather.predict(pool)
p2 = model_weather.predict_proba(pool)
print("class = ", c2)
print("proba = ", p2)
"""

"""
def predict(b_forehead, b_chin, b_cheek, s_forehead, s_chin, s_cheek, h_forehead, h_chin, h_cheek):
    
    cat_tone = ''
    cat_weather = ''
    
    s_data = {
        'b': [b_forehead],
        'b_chin': [b_chin],
        'b_cheek': [b_cheek],
        's': [s_forehead],
        's_chin': [s_chin],
        's_cheek': [s_cheek],
        'h': [h_forehead],
        'h_chin': [h_chin],
        'h_cheek': [h_cheek]
    }
    
    df = pd.DataFrame(s_data)
    
    pool = Pool(df)
    
    c1 = model_tone.predict(pool)
    p1 = model_tone.predict_proba(pool)
    #print('c1 = ', c1)
    #print('p1 = ', p1)
    
    c2 = model_weather.predict(pool)
    p2 = model_weather.predict_proba(pool)
    #print("class = ", c2)
    #print("proba = ", p2)
    
    if c1[0] == 'warm':
        cat_tone = '웜'
    elif c1[0] == 'cool':
        cat_tone = '쿨'
    
    if c2[0][0] == 'spring':
        cat_weather = '봄'
    elif c2[0][0] == 'summer':
        cat_weather = '여름'
    elif c2[0][0] == 'fall':
        cat_weatehr = '가을'
    elif c2[0][0] == 'winter':
        cat_weather = '겨울'
    
    
    return c1[0], c2[0][0]
"""
    