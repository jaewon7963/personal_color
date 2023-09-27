import numpy as np


def cvtcolor2(skin_color):
    
    test_color = np.array([skin_color[2], skin_color[1], skin_color[0]]) / 255
    
    v = 0
    h = 0
    s = 0
    
    
    r = test_color[0]
    g = test_color[1]
    b = test_color[2]
    
    v_min = min( r, g, b )
    v_max = max( r, g, b )
    d_max = v_max - v_min
    
    v = v_max
    
    if d_max == 0:
        h = 0
        s = 0
    else:
        s = d_max / v_max
        
        d_r = ( ( ( v_max - r ) / 6 ) + ( d_max / 2 ) ) / d_max
        d_g = ( ( ( v_max - r ) / 6 ) + ( d_max / 2 ) ) / d_max
        d_b = ( ( ( v_max - r ) / 6 ) + ( d_max / 2 ) ) / d_max
        
        if r == v_max:
            h = d_b - d_g
        elif g == v_max:
            h = ( 1 / 3 ) + d_r - d_b
        elif b == v_max:
            h = ( 2 / 3 ) + d_g - d_r
            
        
        if h < 0 :
            h += 1
            
        if h > 1:
            h -= 1

    return (h,s,v)


def cvtcolor(skin_color):
    

        test_color = np.array([skin_color[2], skin_color[1], skin_color[0]]) / 255
        
        v = 0
        h = 0
        s = 0
        
        
        r = test_color[0]
        g = test_color[1]
        b = test_color[2]
        
        v_min = min( r, g, b )
        v_max = max( r, g, b )
        d_max = v_max - v_min
        
        
        v = v_max * 100
        
        if v_max == 0:
            s = 0
        else:
            s = d_max / v_max * 100
            
        if d_max == 0:
            h = 0
        elif v_max == r:
            h = 60 * ( ((g - b)/d_max) % 6 )
        elif v_max == g:
            h = 60 * ( (b - r)/d_max + 2 )
        elif v_max == b:
            h = 60 * ( (r - g)/d_max + 4 )

        return (h,s,v)
    

print(cvtcolor((255, 0, 255)))