import numpy as np
import cv2

video = cv2.VideoCapture(r"Cars On Highway.mp4")
out = cv2.VideoWriter("Part_1.avi" ,cv2.VideoWriter_fourcc(*'XVID'), 25, (1920, 1080))

# Parameters for lucas kanade optical flow
lucas_para = dict(winSize=(25, 25),
                  maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Creating some random colors
color = np.random.randint(0, 255, (100, 3))

# Taking first frame and find corners in it
ret, frame_old = video.read()
gray_old = cv2.cvtColor(frame_old, cv2.COLOR_BGR2GRAY)

p_old = []
for i in range(0, gray_old.shape[0], 25):
    for j in range(0, gray_old.shape[1], 25):
        p_old.append([[j, i]])
p_old = np.float32(p_old)
f_num = 0
# Creating a making image
mask = np.zeros_like(frame_old)
while (1):
    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculating optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_old, frame_gray, p_old, None, **lucas_para)
    # Selecting good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p_old[st == 1]
    # drawing the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        cv2.arrowedLine(frame, (c,d), (a,b), (0,255,0),3, tipLength=2)
    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow Vector field', img)

    # Updating the previous frame and previous points
    gray_old = frame_gray.copy()
    out.write(img)
    f_num+=1
    if cv2.waitKey(5) & 0xFF == 27:
        break
