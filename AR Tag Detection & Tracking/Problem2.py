import cv2
import numpy as np
import time

# Function for geting information of April tag such as ID and orientation
def detect_id(image):
    orient = ''
    ret, img_binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    img_unpadded = img_binary[50:150, 50:150]
    Test_binaryPoints = np.array([[37, 37], [62, 37], [37, 62], [62, 62]])
    Test_orientPoints = np.array([[85, 85], [15, 85], [15, 15], [85, 15]])
    white = 255
    binarylist = []
    for i in range(0, 4):
        x = Test_binaryPoints[i][0]
        y = Test_binaryPoints[i][1]
        if (img_unpadded[x, y]) == white:
            binarylist.append('1')
        else:
            binarylist.append('0')
    if img_unpadded[Test_orientPoints[0][0], Test_orientPoints[0][1]] == white:
        orient = 3
    elif img_unpadded[Test_orientPoints[1][0], Test_orientPoints[1][1]] == white:
        orient = 2
    elif img_unpadded[Test_orientPoints[2][0], Test_orientPoints[2][1]] == white:
        orient = 1
    elif img_unpadded[Test_orientPoints[3][0], Test_orientPoints[3][1]] == white:
        orient = 0
    returnstring = str(binarylist)
    return returnstring, orient

# Function to detect contours and sorting them
def detect_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    ret, thresh = cv2.threshold(gray, 190, 255, 0)
    all_cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # to exclude the wrong contours
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


# Function to find the homography of fixed square image
def homography(corners, dim=200):
    # Define the eight points to compute the homography matrix
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    # ccw corners
    xp = [0, dim, dim, 0]
    yp = [0, 0, dim, dim]
    n = 9
    m = 8
    A = np.empty([m, n])
    val = 0
    for row in range(0, m):
        if (row % 2) == 0:
            A[row, 0] = -x[val]
            A[row, 1] = -y[val]
            A[row, 2] = -1
            A[row, 3] = 0
            A[row, 4] = 0
            A[row, 5] = 0
            A[row, 6] = x[val] * xp[val]
            A[row, 7] = y[val] * xp[val]
            A[row, 8] = xp[val]

        else:
            A[row, 0] = 0
            A[row, 1] = 0
            A[row, 2] = 0
            A[row, 3] = -x[val]
            A[row, 4] = -y[val]
            A[row, 5] = -1
            A[row, 6] = x[val] * yp[val]
            A[row, 7] = y[val] * yp[val]
            A[row, 8] = yp[val]
            val += 1

    U, S, V = np.linalg.svd(A)
    x = V[-1]
    H = np.reshape(x, [3, 3])
    return H


# Function to apply warp to change field of view
def warp(H, src, h, w):
    idxy, idxx = np.indices((h, w), dtype=np.float32)
    lin_homg_ind = np.array([idxx.ravel(), idxy.ravel(), np.ones_like(idxx).ravel()])

    map_ind = H.dot(lin_homg_ind)
    x_map, y_map = map_ind[:-1] / map_ind[-1]
    x_map = x_map.reshape(h, w).astype(np.float32)
    y_map = y_map.reshape(h, w).astype(np.float32)

    x_map[x_map >= src.shape[1]] = -1
    x_map[x_map < 0] = -1
    y_map[y_map >= src.shape[0]] = -1
    x_map[y_map < 0] = -1

    return_img = np.zeros((h, w, 3), dtype="uint8")
    for x_new in range(w):
        for y_new in range(h):
            x = int(x_map[y_new, x_new])
            y = int(y_map[y_new, x_new])

            if x == -1 or y == -1:
                pass
            else:
                return_img[y_new, x_new] = src[y, x]
    return return_img


def changeOrient(image, orient):
    if orient == 1:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orient == 2:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_180)
    elif orient == 3:
        reoriented_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        reoriented_img = image
    return reoriented_img


def imposetestudo(frame, contour, color):
    cv2.drawContours(frame, [contour], -1, (color), thickness=-1)
    return frame


def project_mat(K, H):
    h1 = H[:, 0]
    h2 = H[:, 1]

    K = np.transpose(K)

    inv_K = np.linalg.inv(K)
    a = np.dot(inv_K, h1)
    c = np.dot(inv_K, h2)
    lamda = 1 / ((np.linalg.norm(a) + np.linalg.norm(c)) / 2)

    B_T = np.dot(inv_K, H)

    if np.linalg.det(B_T) > 0:
        B = 1 * B_T
    else:
        B = -1 * B_T

    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    r1 = lamda * b1
    r2 = lamda * b2
    r3 = np.cross(r1, r2)
    t = lamda * b3
    P = np.dot(K, (np.stack((r1, r2, r3, t), axis=1)))

    return P


def cubepts(corners, H, P, dim):
    new_corners = []
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    s1 = np.stack((np.array(x), np.array(y), np.ones(len(x))))
    s2 = np.dot(H, s1)
    s3 = s2 / s2[2]

    P_w = np.stack((s3[0], s3[1], np.full(4, -dim), np.ones(4)), axis=0)

    sP_c = np.dot(P, P_w)
    P_c = sP_c / (sP_c[2])
    for i in range(4):
        new_corners.append([int(P_c[0][i]), int(P_c[1][i])])

    return new_corners


def drawCube(tagcorners, new_corners, frame, flag):
    thickness = 3
    if not flag:
        contours = makeContours(tagcorners, new_corners)
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), thickness=-1)

    for i, point in enumerate(tagcorners):
        cv2.line(frame, tuple(point), tuple(new_corners[i]), (0, 0, 255), thickness)

    for i in range(4):
        if i == 3:
            cv2.line(frame, tuple(tagcorners[i]), tuple(tagcorners[0]), (0, 255, 0), thickness)
            cv2.line(frame, tuple(new_corners[i]), tuple(new_corners[0]), (0, 255, 0), thickness)
        else:
            cv2.line(frame, tuple(tagcorners[i]), tuple(tagcorners[i + 1]), (0, 255, 0), thickness)
            cv2.line(frame, tuple(new_corners[i]), tuple(new_corners[i + 1]), (0, 255, 0), thickness)

    return frame


def makeContours(corners1, corners2):
    contours = []
    for i in range(len(corners1)):
        if i == 3:
            p1 = corners1[i]
            p2 = corners1[0]
            p3 = corners2[0]
            p4 = corners2[i]
        else:
            p1 = corners1[i]
            p2 = corners1[i + 1]
            p3 = corners2[i + 1]
            p4 = corners2[i]
        contours.append(np.array([p1, p2, p3, p4], dtype=np.int32))
    contours.append(np.array([corners1[0], corners1[1], corners1[2], corners1[3]], dtype=np.int32))
    contours.append(np.array([corners2[0], corners2[1], corners2[2], corners2[3]], dtype=np.int32))
    
    return contours


def getCorners(frame):
    [tag_cnts, corners] = detect_contour(frame, 180)
    tag_corners = {}

    for i, tag in enumerate(corners):
        # compute homography
        size = 200
        H = homography(tag, size)
        H_inv = np.linalg.inv(H)

        # get squared tile
        square_img = warp(H_inv, frame, size, size)
        imgray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        ret, square_img = cv2.threshold(imgray, 180, 255, cv2.THRESH_BINARY)

        # encode squared tile
        [id_str, orientation] = detect_id(square_img)

        order_corners = []

        if orientation == 0:
            order_corners = tag

        elif orientation == 1:
            order_corners.append(tag[1])
            order_corners.append(tag[2])
            order_corners.append(tag[3])
            order_corners.append(tag[0])

        elif orientation == 2:
            order_corners.append(tag[2])
            order_corners.append(tag[3])
            order_corners.append(tag[0])
            order_corners.append(tag[1])

        elif orientation == 3:
            order_corners.append(tag[3])
            order_corners.append(tag[0])
            order_corners.append(tag[1])
            order_corners.append(tag[2])

        tag_corners[id_str] = order_corners

    return tag_corners


def getTopCorners(bot_corners):
    K = np.array([[1406.08415449821, 0, 0],
                  [2.20679787308599, 1417.99930662800, 0],
                  [1014.13643417416, 566.347754321696, 1]])

    top_corners = {}

    for tag_id, corners in bot_corners.items():
        H = homography(corners, 200)
        H_inv = np.linalg.inv(H)
        P = project_mat(K, H_inv)
        top_corners[tag_id] = cubepts(corners, H, P, 200)

    return top_corners


def avgCorners(past, current, future):
    diff = 50
    average_corners = {}
    for tag in current:
        templist = [current[tag]]
        if past == []:
            pass
        elif tag in past[-1]:
            for d in past:
                if tag in d:
                    templist.append(d[tag])
        else:
            pass

        if tag in future[0]:
            for d in future:
                if tag in d:
                    templist.append(d[tag])
        else:
            pass

        newcorners = []
        c1x = c1y = c2x = c2y = c3x = c3y = c4x = c4y = 0

        for allcorners in templist:
            c1x += allcorners[0][0]
            c1y += allcorners[0][1]
            c2x += allcorners[1][0]
            c2y += allcorners[1][1]
            c3x += allcorners[2][0]
            c3y += allcorners[2][1]
            c4x += allcorners[3][0]
            c4y += allcorners[3][1]

        newcorners = np.array([[c1x, c1y], [c2x, c2y], [c3x, c3y], [c4x, c4y]])
        newcorners = np.divide(newcorners, len(templist))
        newcorners = newcorners.astype(int)
        newcorners = np.ndarray.tolist(newcorners)

        # If any coner value is > n pixels from original keep original
        teleport = False
        for i in range(4):
            orig_x = current[tag][i][0]
            orig_y = current[tag][i][1]
            x_new = newcorners[i][0]
            y_new = newcorners[i][1]
            if (abs(orig_x - x_new) > diff) or (abs(orig_y - y_new) > diff):
                teleport = True
        if teleport:
            average_corners[tag] = current[tag]
        else:
            average_corners[tag] = newcorners

    return average_corners


# All Necessary initalization
f_testudo = False
f_cube = False
img_testudo = cv2.imread('testudo.png')
tag_ids = ['0101', '0111', '1111']

# User Inputs
tag_name = input("Input tag file(without extension): ")
print("Choose what to show:\n Enter 1 for Testudo\n Enter 2 for Cube")
i = int(input("Choice: "))
if i == 1:
    f_testudo = True
elif i == 2:
    f_cube = True
else:
    print("Input error.")
    exit(0)

# Capturing video according to the input
video = cv2.VideoCapture(str(tag_name) + '.mp4')

video_encode = cv2.VideoWriter_fourcc(*'XVID')
date = time.strftime("%m-%d ")
vid_name = str(date) + str(tag_name) + ('Testudo' if f_testudo == True else 'cube')
frame_rate = 20
output_video = cv2.VideoWriter(str(vid_name) + ".avi", video_encode, frame_rate, (1920, 1080))

# Camera intrinsic parameters
K = np.array([[1406.08415449821, 0, 0],
              [2.20679787308599, 1417.99930662800, 0],
              [1014.13643417416, 566.347754321696, 1]])
start_frame = 0
count = start_frame
video.set(1, start_frame)

while video.isOpened():
    ret, frame = video.read()
    [all_cnts, cnts] = detect_contour(frame)  # Detect Contours
    cv2.drawContours(frame, all_cnts, -1, (255, 0, 0), 4)  # Draw Detected Contours
    for i, tag in enumerate(cnts):
        H = homography(tag)         # finding Homography
        H_inv = np.linalg.inv(H)
        img_square = warp(H_inv, frame, 200, 200)       # Warping image
        img_gray = cv2.cvtColor(img_square, cv2.COLOR_BGR2GRAY)
        ret, img_square = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)

        [str_id, orient] = detect_id(img_square)  # Detecting orientation and tag_id from image

        if f_testudo:
            img_n = img_testudo
            img_rotate = changeOrient(img_n, orient)
            size = img_rotate.shape[0]
            H = homography(tag, size)
            h = frame.shape[0]
            w = frame.shape[1]
            # Superimposing 'testudo.png' on the tag with given orientation
            frame1 = warp(H, img_rotate, h, w)
            frame2 = imposetestudo(frame, all_cnts[i], 0)
            superimposed_frame = cv2.bitwise_or(frame1, frame2)
            cv2.imshow("Testudo", superimposed_frame)
            output_video.write(superimposed_frame)

        # Generate and impose an Augmented 3d Cube on the tag
        if f_cube:
            H = homography(tag, 200)
            H_inv = np.linalg.inv(H)
            P = project_mat(K, H_inv)
            new_corners = cubepts(tag, H, P, 200)
            frame = drawCube(tag, new_corners, frame, 0)

            cv2.imshow("3D cube", frame)
            output_video.write(frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
