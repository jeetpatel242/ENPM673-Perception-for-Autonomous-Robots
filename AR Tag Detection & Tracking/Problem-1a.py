import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Tag1.mp4')     # capturing video from the file

# Function to detect contours and sorting them
def contour_detection(frame):
    img_blur = cv2.medianBlur(frame, 5)
    ret, thresh = cv2.threshold(img_blur, 190, 255, 0)
    all_cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    wrong_cnts = []
    for i, h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            wrong_cnts.append(i)
    cnts = [c for i, c in enumerate(all_cnts) if i not in wrong_cnts]

    # sort the contours to include only the three largest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    return_cnts = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri * .015, True)
        if len(approx) == 4:
            return_cnts.append(approx)

    corners = []
    for shape in return_cnts:
        points = []
        for p in shape:
            points.append([p[0][0], p[0][1]])
        corners.append(points)

    return return_cnts, corners

# setting the single frame
cap.set(1, 100)
is_success, image = cap.read()

# converting into the greyscale image
imggray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, img_threshold = cv2.threshold(imggray, 150, 255, cv2.THRESH_BINARY)

#  Applying FFT to detect edges
rows, cols = imggray.shape
crow, ccol = int(rows / 2), int(cols / 2)  # center

# Circular HPF mask, center circle is 0, remaining all ones
mask = np.ones((rows, cols, 2), np.uint8)
r = 80
center = [crow, ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
mask[mask_area] = 1
dft = cv2.dft(np.float32(img_threshold), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# applying the mask and inverse DFT
fshift = dft_shift * mask
fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
img_back = np.array(img_back)

# scaling
pmin = img_back.min()
pmax = img_back.max()
a = (255) / (pmax - pmin)
b = (255) - a * pmax
new_img = (a * img_back + b).astype('uint8')

# calling function to find contours and draw
[all_cnts, cnts] = contour_detection(new_img)
cv2.drawContours(image,all_cnts,-1,(0,255,0), 4)

# Plotting the graphs
plt.subplot(1, 5, 1), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 5, 2), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 5, 3), plt.imshow(fshift_mask_mag, cmap='gray')
plt.title('FFT + Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 5, 4), plt.imshow(img_back, cmap='gray')
plt.title('After FFT Inverse'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 5, 5), plt.imshow(image, cmap='gray')
plt.title('Contour'), plt.xticks([]), plt.yticks([])

# Showing the final figure
plt.show()

cv2.imshow("contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
