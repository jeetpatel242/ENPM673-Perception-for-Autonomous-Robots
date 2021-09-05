import cv2
import numpy as np


# Capturing Video
cap = cv2.VideoCapture(r"Cars On Highway.mp4")
out = cv2.VideoWriter("Part_2.avi" ,cv2.VideoWriter_fourcc(*'XVID'), 25, (1920, 1080))
# Capturing the first frame
ret, first_frame = cap.read()

# Converting frame to grayscale because we only need the luminance channel for detecting edges
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Creating an image filled with zeroes
mask = np.zeros_like(first_frame)

# Setting image saturation to maximum
mask[..., 1] = 255

f_num = 0
while(cap.isOpened()):
	ret, frame = cap.read()
	# Converts each frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Calculating dense optical flow by Farneback method
	flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	
	# Computing the magnitude and angle of the 2D vectors
	magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	
	# Setting image hue according to the optical flow
	mask[..., 0] = angle * 180 / np.pi / 2
	
	# Sets image magnitude (normalized) according to the optical flow
	mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
	
	# Converting HSV to RGB (BGR) color representation
	rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

	cv2.imshow("Dense optical flow", rgb)
	out.write(rgb)
	# Updating previous frame
	prev_gray = gray
	f_num += 1
	if cv2.waitKey(5) & 0xFF == 27:
		break

cap.release()
cv2.destroyAllWindows()
