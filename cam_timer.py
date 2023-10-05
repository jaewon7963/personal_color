import cv2 as cv
import time
 
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 5
font_color = (255, 60, 60)
font_thickness = 5

def capture():
    cap = cv.VideoCapture(0)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    

    timer_duration = 5 
    start_time = time.time()
    pic = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        remaining_time = int(timer_duration - (current_time - start_time))
        if remaining_time >= 1:
            cv.putText(frame, f"{remaining_time}", (int(frame_height/2), int(frame_width/2)), font, font_scale, font_color, font_thickness)
        
        cv.imshow('Camera', frame)

        if remaining_time <= 0:
            pic = frame
            break
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    
    return pic
