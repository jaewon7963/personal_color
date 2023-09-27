import numpy as np

def function_v(val):
    if val > 0.04045:
        return pow ( ( (val+0.055) / 1.055) , 2.4)
    else:
        return val / 12.92
    
def function_l(val):
    if val > 0.008856:
        return pow( val , 1/3 )
    else:
        return (7.787 * val) + (16 / 116)
    
    
def cvtcolor(skin_color):    

    np_matrix = np.array( [[0.412, 0.3576, 0.1805],
                           [0.2126, 0.7152, 0.0722],
                           [0.0193, 0.1192, 0.9505]] )
    
    test_color = np.array([skin_color[2], skin_color[1], skin_color[0]]) / 255
    
    r = function_v(test_color[0])
    g = function_v(test_color[1])
    b = function_v(test_color[2])
        
        
    val_r = r * 100
    val_g = g * 100
    val_b = b * 100
    
    rgb = np.array( [val_r, val_g, val_b] )
    
    xyz = np_matrix @ rgb
    
    val_x = xyz[0] / 95.01
    val_y = xyz[1] / 100.0
    val_z = xyz[2] / 108.9
    
    
    val_x = function_l(val_x)
    val_y = function_l(val_y)
    val_z = function_l(val_z)
    
    l = (116 * val_y) - 16
    a = 500 * ( val_x - val_y )
    b = 200 * ( val_y - val_z )