import cv2
import numpy as np
import time


def adjust_gamma(img_resize,gamma):
    #Scale the input from (0 to 255) to (0 to 1)
    # Output = Input^(1/gamma)
    #Apply gamma correction
    #Scale back to original values
    gamma = 1/gamma
    lookup_t =[]
    for i in np.arange(0,256).astype(np.uint8):
        lookup_t.append(np.uint8(((i/255)**gamma)*255))
    lookup = np.array(lookup_t)
    #Creating the lookup table, cv can find the gamma corrections value of each pixel value
    corrections = cv2.LUT(img_resize,lookup)
    return corrections


video = cv2.VideoCapture('Night Drive - 2689.mp4')
# video.set(1,350); # Where frame_no is the frame you want
# ret, frame = video.read() # Read the frame
# cv2.imshow('window_name', frame) # show frame on window

print("Writing the video...")
if video.isOpened() == False:
    print('Input Error!')

date = time.strftime("%m-%d ")
vid_name = str(date) + " " + "Enhance_video.avi"    
out = cv2.VideoWriter(vid_name ,cv2.VideoWriter_fourcc(*'XVID'), 25, (960, 540))

f_num = 0
while video.isOpened():
    success, frame = video.read()
    if success == False:
        break
    img_resize = cv2.resize(frame, (0,0),fx=0.5,fy=0.5)
    img_gamma = adjust_gamma(img_resize, 2)
    img_sharpen = cv2.convertScaleAbs(img_gamma,-1,0.8,3)
    out.write(img_sharpen)
  
    f_num+=1
    if cv2.waitKey(5) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()