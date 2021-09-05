import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to preprocess the input images
def read_images(img1_path, img2_path):
    # Load an color image in grayscale
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    # Convert image to array
    img1_array = np.asarray(img1)
    img2_array = np.asarray(img2)
    print(img1_array.shape)
    print(img2_array.shape)
    return img1_array, img2_array


# Function to compute the fundamental matrix by computing the SVD of Ax = 0
# 8-point algorithm
def get_fundamental_matrix(f1, f2):
    # computing the centroids
    f1_meanX = np.mean(f1[:, 0])
    f1_meanY = np.mean(f1[:, 1])
    f2_meanX = np.mean(f2[:, 0])
    f2_meanY = np.mean(f2[:, 1])
    # Recentering the coordinates by subtracting from mean
    f1[:, 0] = f1[:, 0] - f1_meanX
    f1[:, 1] = f1[:, 1] - f1_meanY
    f2[:, 0] = f2[:, 0] - f2_meanX
    f2[:, 1] = f2[:, 1] - f2_meanY
    s1 = np.sqrt(2.) / np.mean(np.sum(f1 ** 2, axis=1) ** (1 / 2))
    s2 = np.sqrt(2.) / np.mean(np.sum(f2 ** 2, axis=1) ** (1 / 2))

    # Calculating the transformation matrices
    Ta_1 = np.array([[s1, 0, 0], [0, s1, 0], [0, 0, 1]])
    Ta_2 = np.array([[1, 0, -f1_meanX], [0, 1, -f1_meanY], [0, 0, 1]])
    Ta = Ta_1 @ Ta_2
    Tb_1 = np.array([[s2, 0, 0], [0, s2, 0], [0, 0, 1]])
    Tb_2 = np.array([[1, 0, -f2_meanX], [0, 1, -f2_meanY], [0, 0, 1]])
    Tb = Tb_1 @ Tb_2
    # Compute the normalized point correspondences
    x1 = (f1[:, 0].reshape((-1, 1))) * s1
    y1 = (f1[:, 1].reshape((-1, 1))) * s1
    x2 = (f2[:, 0].reshape((-1, 1))) * s2
    y2 = (f2[:, 1].reshape((-1, 1))) * s2

    A = np.hstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones((len(x1), 1))))
    # Solve for A using SVD
    A = np.array(A)
    u, s, v = np.linalg.svd(A)
    v = v.T
    # last column = solution
    sol = v[:, -1]
    F = sol.reshape((3, 3))
    u_f, s_f, v_f = np.linalg.svd(F)
    # Rank-2 constraint
    s_f[2] = 0
    S_new = np.diag(s_f)
    # Recompute normalized F
    f_new = u_f @ S_new @ v_f
    f_norm = Tb.T @ f_new @ Ta
    f_norm = f_norm / f_norm[-1, -1]
    return f_norm


# Function to implement RANSAC algorithm to find the inliers and the best fundamental matrix
def fundamental_matrix_estimation(feature1, feature2):
    thresh = 0.5
    p = 0.99
    BEST_f_matrix = []
    max_num_inliers = 0
    N = np.inf
    count = 0
    while count < N:
        f1_rand = []
        f2_rand = []
        random = np.random.randint(len(feature1), size=8)
        for i in random:
            f1_rand.append(feature1[i])
            f2_rand.append(feature2[i])
        F = get_fundamental_matrix(np.array(f1_rand), np.array(f2_rand))
        ones = np.ones((len(feature1), 1))
        x1 = np.hstack((feature1, ones))
        x2 = np.hstack((feature2, ones))
        s1 = np.dot(x1, F.T)
        s2 = np.dot(x2, F)
        error = np.sum(s2 * x1, axis=1, keepdims=True) ** 2 / np.sum(np.hstack((s1[:, :-1], s2[:, :-1])) ** 2, axis=1,
                                                                     keepdims=True)
        inlier = error <= thresh
        inliers_cnt = np.sum(inlier)

        if inliers_cnt > max_num_inliers:
            max_num_inliers = inliers_cnt
            BEST_f_matrix = F
        inlier_ratio = inliers_cnt / len(feature1)
        if np.log(1 - (inlier_ratio ** 8)) == 0:
            continue
        N = np.log(1 - p) / np.log(1 - (inlier_ratio ** 8))
        count += 1
    return BEST_f_matrix


# Function to compute essential matrix from fundamental matrix
def essential_matrix(F, K):
    E = np.dot(K.T, np.dot(F, K))
    u, s, v = np.linalg.svd(E)
    # correction of singular values by reconstructing it with (1, 1, 0) singular values
    s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    E_new = u @ s_new @ v
    return E_new


# Function to derive the camera position and orientation from given the essential matrix
def extract_camera_pose(E, K):
    u, d, v = np.linalg.svd(E)
    v = v.T
    W = np.reshape([0, -1, 0, 1, 0, 0, 0, 0, 1], (3, 3))

    c1 = u[:, 2]
    c2 = -u[:, 2]
    c3 = u[:, 2]
    c4 = -u[:, 2]
    r1 = u @ W @ v.T
    r2 = u @ W @ v.T
    r3 = u @ W.T @ v.T
    r4 = u @ W.T @ v.T

    if np.linalg.det(r1) < 0:
        r1 = -r1
        c1 = -c1
    if np.linalg.det(r2) < 0:
        r2 = -r2
        c2 = -c2
    if np.linalg.det(r3) < 0:
        r3 = -r3
        c3 = -c3
    if np.linalg.det(r4) < 0:
        r4 = -r4
        c4 = -c4

    c1 = c1.reshape((3, 1))
    c2 = c2.reshape((3, 1))
    c3 = c3.reshape((3, 1))
    c4 = c4.reshape((3, 1))

    return [r1, r2, r3, r4], [c1, c2, c3, c4]


# Function to draw feature points and epipolar lines between two images
def drawMatches(img1, img2, lines, pts1, pts2):
    # drawing epilines from img-1 to img-2
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Function to extract keypoints and descriptors
def sift_detector(imgA, imgB):
    sift = cv2.SIFT_create()
    # Using SIFT to find the keypoints and descriptors 
    kp1, des1 = sift.detectAndCompute(imgA, None)
    kp2, des2 = sift.detectAndCompute(imgB, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # ratio test for best matches
    good = []
    best_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            best_matches.append([m])

    feature_1 = []
    feature_2 = []

    for i, match in enumerate(good):
        feature_1.append(kp1[match.queryIdx].pt)
        feature_2.append(kp2[match.trainIdx].pt)
    return feature_1, feature_2, kp1, kp2, best_matches


# Function to visualize the matching window concept
def display_image(img, window_name='Matching Window'):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function to find the absolute difference between two pixels
def sum_of_abs_diff(pixel1_values, pixel2_values):
    if pixel1_values.shape != pixel2_values.shape:
        return -1
    return np.sum(abs(pixel1_values - pixel2_values))


# Function to compare the block of image-1 with image-2 within block size
def compare_blocks(y, x, block_img2, img2_arrayblock, block_size=5):
    # Getting search range for the right image
    x_min = max(0, x - search_block)
    x_max = min(img2_arrayblock.shape[1], x + search_block)
    min_sad = None
    min_index = None
    first = True
    for x in range(x_min, x_max):
        block_right = img2_arrayblock[y: y + block_size,
                      x: x + block_size]
        sad = sum_of_abs_diff(block_img2, block_right)
        if first:
            min_sad = sad
            min_index = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                min_index = (y, x)

    return min_index


# Function to draw window in image-2
def image2_block(x, y):
    array_img1, array_img2 = read_images(img1_path, img2_path)
    img_2 = cv2.imread("dataset 2/im1.png", 0)
    x_min = max(0, x - 50)
    x_max = min(array_img2.shape[1], x + 25)
    img1_bbox = cv2.rectangle(img_2, (x_min, y),
                                  (x_max, y + 25),
                                  (0, 0, 255), 2)
    display_image(img1_bbox, window_name='right')


# Function to get disparity map
def plot_disparity_map():
    imgA_array, imgB_array = read_images(img1_path, img2_path)
    imgA_array = imgA_array.astype(int)
    imgB_array = imgB_array.astype(int)
    if imgA_array.shape != imgB_array.shape:
        raise ("img1 and img2 shape mismatch!")
    h, w = imgA_array.shape
    map_disparity = np.zeros((h, w))
    # Going over each pixel position
    print('Loading...It will take about an hour...')
    for y in range(block_size, h - block_size):
        print(y+1, end=" ", flush=True)
        for x in range(block_size, w - block_size):
            block_left = imgA_array[y:y + block_size,
                         x:x + block_size]
            min_index = compare_blocks(y, x, block_left,
                                       imgB_array,
                                       block_size=block_size)
            map_disparity[y, x] = abs(min_index[1] - x)
    plt.imshow(map_disparity, cmap='hot', interpolation='nearest')
    plt.savefig('disparity_image.png')
    plt.show()
    plt.imshow(map_disparity, cmap='gray', interpolation='nearest')
    plt.savefig('disparity_image_gray.png')
    plt.show()
    return map_disparity


# Function to get depth map of image-1 and image-2
def plot_depth_map(map_disp, B, f):
    imgA_array, imgB_array = read_images(img1_path, img2_path)
    imgA_array = imgA_array.astype(int)
    imgB_array = imgB_array.astype(int)
    if imgA_array.shape != imgB_array.shape:
        raise ("img1 and img2 shape mismatch!")
    h, w = imgA_array.shape
    map_depth = np.zeros((h, w))
    h, w = map_disp.shape
    for i in range(h):
        for j in range(w):
            map_depth[i, j] = (B * f) / map_disp[i, j]
    plt.imshow(map_depth, cmap='hot', interpolation='nearest')
    plt.savefig('deapth_image.png')
    plt.show()
    plt.imshow(map_depth, cmap='gray', interpolation='nearest')
    plt.savefig('deapth_image_gray.png')
    plt.show()


#  ********** Main ************

# Taking input from the user
input_set = int(input("Choose the dataset (1 or 2 or 3): "))
if input_set == 1:
    img1_path = "Dataset 1/im0.png"
    img2_path = "Dataset 1/im1.png"
    img1 = cv2.imread('Dataset 1/im0.png')
    img2 = cv2.imread('Dataset 1/im1.png')
    c1 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
    c2 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
    baseline = 177.28
    focal = 5299.313

elif input_set == 2:
    img1_path = "Dataset 2/im0.png"
    img2_path = "Dataset 2/im1.png"
    img1 = cv2.imread('Dataset 2/im0.png')
    img2 = cv2.imread('Dataset 2/im1.png')
    c1 = np.array([[4396.869, 0, 1353.072], [0, 4396.869, 989.702], [0, 0, 1]])
    c2 = np.array([[4396.869, 0, 1538.86], [0, 4396.869, 989.702], [0, 0, 1]])
    baseline = 144.049
    focal = 4396.869

elif input_set == 3:
    img1_path = "Dataset 3/im0.png"
    img2_path = "Dataset 3/im1.png"
    img1 = cv2.imread('Dataset 3/im0.png')
    img2 = cv2.imread('Dataset 3/im1.png')
    c1 = np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]])
    c2 = np.array([[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]])
    baseline = 174.490
    focal = 5806.559

else:
    print("Invalid Input")

input_2 = int(input("Choose the function 1. Stereo Rectification, 2. Disparity and depth map: "))
if input_2 == 1:
    h1 = img1.shape[0]
    w1 = img1.shape[1]
    ch1 = img1.shape[2]

    h2 = img2.shape[0]
    w2 = img2.shape[1]
    ch2 = img2.shape[2]

    feat_1, feat_2, kp1, kp2, best_match = sift_detector(img1, img2)
    Best_Fmatrix = fundamental_matrix_estimation(feat_1, feat_2)
    # print("Best Fundamental Matrix:", Best_Fmatrix)

    mat_E = essential_matrix(Best_Fmatrix, c1)
    R, T = extract_camera_pose(mat_E, c1)
    H = []
    I = np.array([0, 0, 0, 1])
    for i, j in zip(R, T):
        h = np.hstack((i, j))
        h = np.vstack((h, I))
        H.append(h)
    print('H:\n', H)
    # Sterio Rectification for horizontal epipolar lines
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(feat_1), np.float32(feat_2), Best_Fmatrix, imgSize=(w1, h1))
    print("H1:\n", H1)
    print("H2:\n", H2)
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    distances = {}
    for j in range(len(feat_1)):
        list_n = np.array([feat_1[j][0], feat_2[j][1], 1])
        list_n = np.reshape(list_n, (3, 1))
        list_n2 = np.array([feat_1[j][0], feat_2[j][1], 1])
        list_n2 = np.reshape(list_n2, (1, 3))
        distances[j] = abs(list_n2 @ Best_Fmatrix @ list_n)

    sort_dist = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    dist = []
    v_list = []
    for k, v in sort_dist.items():
        # threshold distance
        if v[0][0] < 0.05:
            dist.append(v[0][0])
            v_list.append(k)

    len_dist = len(dist)
    len_dist = min(len_dist, 30)
    inlier1 = []
    inlier2 = []
    for x in range(len_dist):
        inlier1.append(feat_1[v_list[x]])
        inlier2.append(feat_2[v_list[x]])

    Match_img = cv2.drawMatchesKnn(img1_rectified, kp1, img2_rectified, kp2, best_match, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(Match_img)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif input_2 == 2:
    block_size = 7
    search_block = 50
    map_disp = plot_disparity_map()
    plot_depth_map(map_disp, baseline, focal)

else:
    print("Invalid Input")