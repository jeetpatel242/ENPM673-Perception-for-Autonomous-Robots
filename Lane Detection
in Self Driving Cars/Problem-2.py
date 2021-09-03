import cv2
import numpy as np
import os
from copy import deepcopy
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Taking input from the user
def GetUserInput():
    for input_set in range(2):
        input_set = int(input("Choose the dataset for lane detection with turn prediction (1 or 2):"))
        if input_set == 1:
            print("Playing the output video of dataset-1. Loading...")
            detect_lane(1)
            break
        elif input_set == 2:
            print("Playing the output video of dataset-2. Loading...")
            detect_lane(2)
            break
        else:
            print("Invalid input entered. Quitting...")
            exit()


# Function to do binary thresholding
def thresh_bin(img, min, max):
    binx = np.zeros_like(img)
    binx[(img >= min) & (img <= max)] = 1
    return binx


# Function for segmenting white and yellow lanes
def color_segment(img_y, data_set, mask1):
    # apply HLS thresholding for yellow line
    hls = cv2.cvtColor(img_y, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    h_bin = thresh_bin(H, 10, 70)
    l_bin = thresh_bin(L, 120, 200)
    s_bin = thresh_bin(S, 120, 255)

    l_yellow = np.zeros_like(img_y[:, :, 0])
    l_yellow[(h_bin == 1) & (l_bin == 1) & (s_bin == 1)] = 1

    # apply RGB thresholding for white line
    R = img_y[:, :, 2]
    G = img_y[:, :, 1]
    B = img_y[:, :, 0]
    r_bin = thresh_bin(R, 200, 255)
    g_bin = thresh_bin(G, 200, 255)
    b_bin = thresh_bin(B, 200, 255)

    l_white = np.zeros_like(img_y[:, :, 0])
    l_white[(r_bin == 1) & (g_bin == 1) & (b_bin == 1)] = 1

    # apply YUV thresholding for illumination
    img_yuv = cv2.cvtColor(img_y, cv2.COLOR_BGR2YUV)
    u_channel = img_yuv[:, :, 1]
    u_bin = thresh_bin(u_channel, 0, 120)

    combine_frames = np.zeros_like(img_y[:, :, 0])
    if data_set == 1:
        combine_frames[((l_white == 1) | ((l_yellow == 1) | (u_bin == 1))) & (mask1 == 1)] = 1
    else:
        combine_frames[((l_white == 1) | (l_yellow == 1)) & (mask1 == 1)] = 1
    return combine_frames


def detect_lane(input_set):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(current_dir, "data_1/data")
    frames = []
    frame_num = []
    input_set = input_set - 1

    if input_set == 1:
        cap = cv2.VideoCapture('data_2/challenge_video.mp4')
        K = np.asarray([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],
                        [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.asarray([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])
        if (cap.isOpened() == False):
            print("Unable to read input from camera")
        while (True):
            success, img_read = cap.read()
            if success:
                img_read = cv2.undistort(img_read, K, dist)
                img_yuv = cv2.cvtColor(img_read, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                img_read = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                img_read = cv2.GaussianBlur(img_read, (5, 5), 5)
                frames.append(img_read)
            else:
                cap.release()
                break

        corners = np.array([[(280, 690), (1100, 690), (760, 480), (600, 480)]])

        # Define Region of interest
        initial = np.float32([[280, 700], [1100, 700], [600, 480], [760, 480]])
        final = np.float32([[0, 400], [200, 400], [0, 0], [200, 0]])

        mask = np.zeros_like(frames[0])
        cv2.fillPoly(mask, corners, (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = thresh_bin(mask, 200, 255)

        # Creating the output video file
        date = time.strftime("%m-%d ")
        vid_name = str(date) + " " + "Output-2.avi"
        video_out = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'DIVX'), 29.97,
                              (frames[0].shape[1], frames[0].shape[0]))

    else:
        K = np.asarray([[9.037596e+02, 0.000000e+00, 6.957519e+02],
                        [0.000000e+00, 9.019653e+02, 2.242509e+02],
                        [0.000000e+00, 0.000000e+00, 1.000000e+00]])

        dist = np.asarray([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
        for file in sorted(os.listdir(img_dir)):
            img_read = cv2.imread(os.path.join(img_dir, file))
            img_read = cv2.undistort(img_read, K, dist)
            img_read = cv2.GaussianBlur(img_read, (5, 5), 20.0)
            if img_read is not None:
                frames.append(img_read)
                frame_num.append(file)
            else:
                print("None")

        corners = np.array([[(150, 495), (950, 495), (740, 280), (530, 280)]])
        mask = np.zeros_like(frames[0])
        cv2.fillPoly(mask, corners, (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = thresh_bin(mask, 200, 255)

        # points for considering region of Interest
        initial = np.float32([[150, 495], [950, 495], [530, 280], [740, 280]])
        final = np.float32([[0, 400], [200, 400], [0, 0], [200, 0]])

        # Creating the output video file
        date = time.strftime("%m-%d ")
        vid_name = str(date) + " " + "Output-1.avi"
        video_out = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'DIVX'), 15,
                                    (frames[0].shape[1], frames[0].shape[0]))

    for i in range(len(frames)):
        img_frame = frames[i]
        img = deepcopy(img_frame)

        # apply hls thresholding for yellow line
        img_cs = color_segment(img, input_set, mask)
        img_transform = cv2.getPerspectiveTransform(initial, final)
        img_bview = cv2.warpPerspective(img_cs, img_transform, (200, 400))
        hist = np.sum(img_bview, axis=0)
        img_out = np.dstack((img_bview, img_bview, img_bview)) * 255
        point_mid = np.int(hist.shape[0] / 2)
        l_basex = np.argmax(hist[:point_mid])
        r_basex = np.argmax(hist[point_mid:]) + point_mid

        win_num = 20
        # Setting height of windows
        win_height = np.int(img_bview.shape[0] / win_num)
        # Identifying the positions of all non-zero images
        nonzero = img_bview.nonzero()
        y_nonzero = np.array(nonzero[0])
        x_nonzero = np.array(nonzero[1])

        # Updating current position for each window
        cur_leftx = l_basex
        cur_lefty = r_basex

        # Creating empty lists to receive left and right lane pixel indices
        list_llane = []
        list_rlane = []

        pixel_min = 50
        # Checking for lane candidates through each windows
        for window in range(win_num):
            ylow = img_bview.shape[0] - (window + 1) * win_height
            yhigh = img_bview.shape[0] - window * win_height
            xleft_low = cur_leftx - 35
            xleft_high = cur_leftx + 35
            xright_low = cur_lefty - 35
            xright_high = cur_lefty + 35
            win_nonzero_l = ((y_nonzero >= ylow) & (y_nonzero < yhigh) & (x_nonzero >= xleft_low ) & (
                        x_nonzero < xleft_high)).nonzero()[0]
            win_nonzero_r = ((y_nonzero >= ylow) & (y_nonzero < yhigh) & (x_nonzero >= xright_low ) & (
                        x_nonzero < xright_high)).nonzero()[0]
            list_llane.append(win_nonzero_l)
            list_rlane.append(win_nonzero_r)
            if len(win_nonzero_l) > pixel_min:
                cur_leftx = np.int(np.mean(x_nonzero[win_nonzero_l]))
            if len(win_nonzero_r) > pixel_min:
                cur_lefty = np.int(np.mean(x_nonzero[win_nonzero_r]))

        # Concatenating the arrays of indices
        list_llane = np.concatenate(list_llane)
        list_rlane = np.concatenate(list_rlane)
        # Extracting the left and right line pixel positions
        x_left = x_nonzero[list_llane]
        y_left = y_nonzero[list_llane]
        x_right = x_nonzero[list_rlane]
        y_right = y_nonzero[list_rlane]

        # Generating curve to fit the lanes
        l_linefit = np.polyfit(y_left, x_left, 2)
        r_linefit = np.polyfit(y_right, x_right, 2)

        p_y = np.linspace(0, img_bview.shape[0] - 1, img_bview.shape[0])
        l_fitx = l_linefit[0] * p_y ** 2 + l_linefit[1] * p_y + l_linefit[2]
        r_fitx = r_linefit[0] * p_y ** 2 + r_linefit[1] * p_y + r_linefit[2]
        # Creating polygon on image
        for i in range(len(p_y)):
            cv2.circle(img_out, (int(l_fitx[i]), int(p_y[i])), 5, (0, 255, 255), -1)
            cv2.circle(img_out, (int(r_fitx[i]), int(p_y[i])), 5, (0, 255, 255), -1)

        pts_left = np.array([np.transpose(np.vstack([l_fitx, p_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r_fitx, p_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(img_out, np.int_([pts]), (255, 0, 255))

        H_inverse = cv2.getPerspectiveTransform(final, initial)
        warp_new = cv2.warpPerspective(img_out, H_inverse, (img.shape[1], img.shape[0]))

        # Overlaying polygon onto the original image
        img = cv2.addWeighted(img, 1, warp_new, 0.4, 0)

        y_min = 0
        l_line = l_linefit[0] * y_min ** 2 + l_linefit[1] * y_min + l_linefit[2]
        r_line = r_linefit[0] * y_min ** 2 + r_linefit[1] * y_min + r_linefit[2]

        # Finding center of top of polygon
        m_linetop = l_line + (r_line - l_line) / 2

        y_max = img_bview.shape[1] - 1
        l_line = l_linefit[0] * y_max ** 2 + l_linefit[1] * y_max + l_linefit[2]
        r_line = r_linefit[0] * y_max ** 2 + r_linefit[1] * y_max + r_linefit[2]

        # Finding center of bottom of polygon
        m_linebottom = l_line + (r_line - l_line) / 2

        # Calculating the deviation of mid-line to predict turns
        deviation = int(m_linetop) - int(m_linebottom)

        if deviation > 10 :
            cv2.putText(img, 'Turning Right', (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2)
            print("Right")
        elif deviation < -10:
            cv2.putText(img, 'Turning Left', (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2)
            print("Left")
        else:
            cv2.putText(img, 'Going Straight', (60, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2)
            print("Straight")

        cv2.imshow("img_out", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        video_out.write(img)
    video_out.release()


GetUserInput()
