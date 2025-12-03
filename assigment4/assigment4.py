import a4_utils as a4
import cv2, random, math
import numpy as np
from matplotlib import pyplot as plt

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 1 
#a)

def partial_derivatives(image, sigma):
    G = a4.gauss(sigma)
    D = a4.gaussdx(sigma)

    image_x1 = a4.convolve(image, G.T)
    image_x = a4.convolve(image_x1, D)

    image_y1 = a4.convolve(image, G)
    image_y = a4.convolve(image_y1, D.T)

    return image_x, image_y

def second_partial_derivatives(image, sigma):
    image_x, image_y = partial_derivatives(image, sigma)

    G = a4.gauss(sigma)
    D = a4.gaussdx(sigma)

    image_xx1 = a4.convolve(image_x, G.T)
    image_xx = a4.convolve(image_xx1, D)

    image_yy1 = a4.convolve(image_y, G)
    image_yy = a4.convolve(image_yy1, D.T)

    return image_xx, image_yy

def second_partial_mixed_derivatives(image, sigma):
    image_x, image_y = partial_derivatives(image, sigma)

    G = a4.gauss(sigma)
    D = a4.gaussdx(sigma)

    image_xy1 = a4.convolve(image_x, G)
    image_xy = a4.convolve(image_xy1, D.T)

    image_yx1 = a4.convolve(image_y, G.T)
    image_yx = a4.convolve(image_yx1, D)

    return image_xy, image_yx

def gradient_magnitude(image, sigma):
    """
    firstly returns magnitude, second angle
    """
    image_x, image_y = partial_derivatives(image, sigma)
    m = np.sqrt(image_x**2 + image_y**2)
    fi = np.arctan2(image_y, image_x)

    return m, fi

def hessian_points(image, sigma, thresh):
    I_xx, I_yy = second_partial_derivatives(image, sigma)
    I_xy, _ = second_partial_mixed_derivatives(image, sigma)

    #debug
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.imshow(I_xx, cmap='gray')
    # plt.title('I_xx')
    # plt.subplot(1,3,2)
    # plt.imshow(I_yy, cmap='gray')
    # plt.title('I_yy')
    # plt.subplot(1,3,3)
    # plt.imshow(I_xy, cmap='gray')
    # plt.title('I_xy')
    # plt.show()


    det = I_xx*I_yy - I_xy*I_xy
    det = (det - det.min()) / (det.max() - det.min())
    det = np.where(det > thresh, det, 0.0)

    # plt.imshow(det, cmap='gray')
    # plt.title('Hessian determinant')
    # plt.show()

    return det

def non_maxima_suppression(det, box_size=7, thresh=0.0, unique_max=False):
    det = np.where(det > thresh, det, 0.0)
    suppressed = np.zeros_like(det)
    R = box_size // 2
    padded = np.pad(det, R, mode='constant', constant_values=0.0)

    ys, xs = np.nonzero(det)
    for y, x in zip(ys, xs):
        y0 = y + R
        x0 = x + R
        window = padded[y0 - R : y0 + R + 1, x0 - R : x0 + R + 1]
        max_val = np.max(window)
        if det[y, x] == max_val:
            if unique_max:
                # keep only if unique maximum
                if np.sum(window == max_val) == 1:
                    suppressed[y, x] = det[y, x]
            else:
                suppressed[y, x] = det[y, x]
    return suppressed

def plot_points(image, points, color='red', s=10):
    ys, xs = np.nonzero(points)  # get coordinates of detected points
    plt.imshow(image)
    plt.scatter(xs, ys, c=color, s=s, marker='x')   # plot points as red dots


graf_color = cv2.imread("./data/graf/graf_a.jpg", cv2.IMREAD_COLOR)  # BGR
graf_color = cv2.cvtColor(graf_color, cv2.COLOR_BGR2RGB)
graf = cv2.cvtColor(graf_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255

points1_hessian = hessian_points(graf, sigma=1, thresh=0.2)
points1 = non_maxima_suppression(points1_hessian, box_size=7, thresh=0.6, unique_max=True)
points2_hessian = hessian_points(graf, sigma=2, thresh=0.2)
points2 = non_maxima_suppression(points2_hessian, box_size=7, thresh=0.7, unique_max=True)
points3_hessian = hessian_points(graf, sigma=3, thresh=0.2)
points3 = non_maxima_suppression(points3_hessian, box_size=7, thresh=0.55, unique_max=True)


plt.figure(figsize=(15, 8))
plt.suptitle("hessian")
plt.subplot(2, 3, 1)
plt.imshow(points1_hessian, cmap="gray")
plt.title("σ=1")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(points2_hessian, cmap="gray")
plt.title("σ=2")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(points3_hessian, cmap="gray")
plt.title("σ=3")
plt.axis('off')
plt.subplot(2, 3, 4)
plot_points(graf_color, points1)
plt.axis('off')
plt.subplot(2, 3, 5)
plot_points(graf_color, points2)
plt.axis('off')
plt.subplot(2, 3, 6)
plot_points(graf_color, points3)
plt.axis('off')

plt.show()

# QUESTION: What kind of structures in the image are detected by the algorithm? How does the parameter sigma affect the result?
#           We are detecting corners, the greater the sigma the less corners we get -> the smaller one starts to disappear

#b)
def harris(image, sigma, thresh):
    sigma2 = 1.6 * sigma
    alfa = 0.06

    I_x, I_y = partial_derivatives(image, sigma)

    C_xx = a4.convolve((I_x * I_x), a4.gauss(sigma2))
    C_yy = a4.convolve((I_y * I_y), a4.gauss(sigma2))
    C_xy = a4.convolve((I_x * I_y), a4.gauss(sigma2))

    det = C_xx*C_yy - C_xy*C_xy
    trace = C_xx + C_yy
    mask = det - alfa*(trace**2)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    out = np.where(mask > thresh, mask, 0)

    return out

graf_color = cv2.imread("./data/graf/graf_a.jpg", cv2.IMREAD_COLOR)  # BGR
graf_color = cv2.cvtColor(graf_color, cv2.COLOR_BGR2RGB)
graf = cv2.cvtColor(graf_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255

points1_harris = harris(graf, sigma=1, thresh=0.2)
points1 = non_maxima_suppression(points1_harris, box_size=7, thresh=0.45, unique_max=True)
points2_harris = harris(graf, sigma=2, thresh=0.2)
points2 = non_maxima_suppression(points2_harris, box_size=7, thresh=0.65, unique_max=True)
points3_harris = harris(graf, sigma=3, thresh=0.2)
points3 = non_maxima_suppression(points3_harris, box_size=7, thresh=0.55, unique_max=True)


plt.figure(figsize=(15, 8))
plt.suptitle("harris")
plt.subplot(2, 3, 1)
plt.imshow(points1_harris, cmap="gray")
plt.title("σ=1")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(points2_harris, cmap="gray")
plt.title("σ=2")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(points3_harris, cmap="gray")
plt.title("σ=3")
plt.axis('off')
plt.subplot(2, 3, 4)
plot_points(graf_color, points1)
plt.axis('off')
plt.subplot(2, 3, 5)
plot_points(graf_color, points2)
plt.axis('off')
plt.subplot(2, 3, 6)
plot_points(graf_color, points3)
plt.axis('off')

plt.show()


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 2
#a)
def hellinger(a, b):
   return np.sqrt(0.5 * np.sum((np.sqrt(a) - np.sqrt(b))**2))

def find_correspondences(desc1, desc2):
    """
    desc1: descriptors from first image (N1 * D)
    desc2: descriptors from second image (N2 * D)

    Returns list of pairs [a, b] meaning:
    descriptor a from desc1 corresponds to descriptor b from desc2
    """
    correspondences = []

    for i in range(len(desc1)):
        d1 = desc1[i]

        distances = np.array([hellinger(d1, d2) for d2 in desc2])

        # Best match is index of minimum distance
        j = np.argmin(distances)

        correspondences.append([i, j])

    return correspondences


A = cv2.imread("./data/graf/graf_a_small.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
B = cv2.imread("./data/graf/graf_b_small.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

# Detect points
detA = hessian_points(A, sigma=3, thresh=0.2)
ptsA = non_maxima_suppression(detA, box_size=5, thresh=0.5)
detB = hessian_points(B, sigma=3, thresh=0.2)
ptsB = non_maxima_suppression(detB, box_size=5, thresh=0.5)

YA, XA = np.nonzero(ptsA)
YB, XB = np.nonzero(ptsB)
print("Task a) Detected keypoints: A =", len(XA), " B =", len(XB))

# Compute descriptors
descA = a4.simple_descriptors(A, YA, XA, n_bins = 16, window_size = 20)
descB = a4.simple_descriptors(B, YB, XB, n_bins = 16, window_size = 20)

corr = find_correspondences(descA, descB)

# Display result
plt.figure(figsize=(12,5))
plt.title("MATCHES: Hessian keypoints + find_correspondences")
a4.display_matches(A, B, np.column_stack((XA, YA)), np.column_stack((XB, YB)), corr)    #np.column_stack -> Stack 1-D arrays as columns into a 2-D array.


#b)
def find_matches(image_A, image_B, sigma=3, thresh_det=0.1, nms_box=7, unique_max=True, desc_window_size=35, desc_bins=16, descriptor="simple"):
    # detect keypoints
    detA = hessian_points(image_A, sigma=sigma, thresh=thresh_det)
    detB = hessian_points(image_B, sigma=sigma, thresh=thresh_det)

    ptsA = non_maxima_suppression(detA, box_size=nms_box, thresh=0.5, unique_max=unique_max)
    ptsB = non_maxima_suppression(detB, box_size=nms_box, thresh=0.5, unique_max=unique_max)


    YA, XA = np.nonzero(ptsA)
    YB, XB = np.nonzero(ptsB)
    print("Detected keypoints: A =", len(XA), " B =", len(XB))

    if(descriptor == "sift"):
        descA = sift_descriptors(A, YA, XA)
        descB = sift_descriptors(B, YB, XB)
    else:
        descA = a4.simple_descriptors(image_A, YA, XA, n_bins=desc_bins, window_size=desc_window_size, sigma=2)
        descB = a4.simple_descriptors(image_B, YB, XB, n_bins=desc_bins, window_size=desc_window_size, sigma=2)
    
    print("Descriptors shapes:", descA.shape, descB.shape)

    # find best matches both directions
    lr = find_correspondences(descA, descB)  # list of [i, j]
    rl = find_correspondences(descB, descA)  # list of [j, i] from B->A

    # map arrays
    lr_map = np.array([b for _, b in lr], dtype=int)   # for each i (A) -> best j (B)
    rl_map = np.array([a for _, a in rl], dtype=int)   # for each j (B) -> best i (A)

    # symmetric matches: indices i such that rl_map[ lr_map[i] ] == i
    valid = np.where((lr_map < len(rl_map)) & (rl_map[lr_map] == np.arange(len(lr_map))))[0]
    matches = np.stack([valid, lr_map[valid]], axis=1)

    print("Symmetric matches count:", matches.shape[0])
    return matches, (XA, YA), (XB, YB)

A = cv2.imread("./data/graf/graf_a_small.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
B = cv2.imread("./data/graf/graf_b_small.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

matches, (XA,YA), (XB,YB) = find_matches(A, B)
#print("Matches:", len(matches))

plt.figure(figsize=(12,6))
plt.title("MATCHES: Hessian keypoints + simple_descriptor")
a4.display_matches(A, B, np.column_stack((XA, YA)), np.column_stack((XB, YB)), matches)


#c)
def sift_descriptors(image, ys, xs, window_size=16, num_cells=4, num_bins=8, sigma=None):
    cell_size = window_size // num_cells
    if sigma is None:
        sigma = 0.5 * window_size            # SIFT recommended
    
    mag, ang = gradient_magnitude(image, sigma)
    ang = (ang + 2*np.pi) % (2*np.pi)

    # Gauss
    g = cv2.getGaussianKernel(window_size, sigma)
    gaussian_window = g @ g.T                         # 16×16 weights

    # Output descriptors
    descs = []

    half = window_size // 2

    H, W = image.shape

    for y, x in zip(ys, xs):
        # skip keypoints near the border
        if y - half < 0 or y + half >= H or x - half < 0 or x + half >= W:
            descs.append(np.zeros(128))
            continue

        # extract patch
        patch_mag = mag[y-half:y+half, x-half:x+half]
        patch_ang = ang[y-half:y+half, x-half:x+half]

        w = gaussian_window

        # SIFT descriptor array (4x4x8)
        descriptor = np.zeros((num_cells, num_cells, num_bins))

        # Fill the histograms
        for i in range(window_size):
            for j in range(window_size):

                m = patch_mag[i, j] * w[i, j]
                theta = patch_ang[i, j]

                # which cell?
                cell_y = i // cell_size
                cell_x = j // cell_size

                # which orientation bin?
                bin_f = (theta / (2*np.pi)) * num_bins
                bin_i = int(bin_f) % num_bins
                bin_w = bin_f - np.floor(bin_f)

                # soft orientation binning -> distributes the pixel’s contribution between two orientation bins
                descriptor[cell_y, cell_x, bin_i] += m * (1 - bin_w)
                descriptor[cell_y, cell_x, (bin_i+1) % num_bins] += m * bin_w

        # flatten to 128-D
        d = descriptor.flatten()

        # normalization & clipping (standard SIFT)
        norm = np.linalg.norm(d) + 1e-10
        d = d / norm
        # reduces dominance of large gradients.
        d = np.clip(d, 0, 0.2)
        d = d / (np.linalg.norm(d) + 1e-10)

        descs.append(d)

    return np.array(descs)


matches, (XA,YA), (XB,YB) = find_matches(A, B, descriptor="sift")
plt.figure(figsize=(12,6))
plt.title("MATCHES: Hessian keypoints + SIFT descriptor")
a4.display_matches(A, B, np.column_stack((XA, YA)), np.column_stack((XB, YB)), matches)

# 1. Increased robustness to rotation
# 2. Increased robustness to illumination changes
# 3. Better repeatability under viewpoint change
# 4. Stronger matching performance


#d)
# Load video
cap = cv2.VideoCapture("20251125_201258.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)  # downsample for speed
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
out = cv2.VideoWriter("stabilized_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#------------------------

# Read first frame
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Cannot read video")

first_frame = cv2.resize(first_frame, (frame_width, frame_height))
gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Use SIFT for keypoints and descriptors
sift = cv2.SIFT_create()
kp_first, des_first = sift.detectAndCompute(gray_first, None)

#-------------------------

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Keep the first frame as reference
prev_gray = gray_first
prev_kp = kp_first
prev_des = des_first

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints in current frame
    kp, des = sift.detectAndCompute(gray, None)

    # Match keypoints to first frame
    matches = bf.match(prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched coordinates
    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Compute affine transform
    # Use RANSAC for robustness
    M, mask = cv2.estimateAffinePartial2D(from_=dst_pts, to=src_pts, method=cv2.RANSAC)

    # Apply transformation to stabilize
    stabilized_frame = cv2.warpAffine(src=frame, M=M, dsize=(frame_width, frame_height))

    # Save frame to video
    out.write(stabilized_frame)

    # Optional: show progress
    combined = np.hstack((frame, stabilized_frame))
    cv2.imshow("Original (left) vs Stabilized (right)", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()




#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 3


# QUESTION: Looking at the equation above, which parameters account for translation and which for rotation and scale?
#           p3, p4 -> translation
#           p1 -> scale
#           p2 -> rotation (+ sheering)
# QUESTION: Write down a sketch of an algorithm to determine similarity transform from a set of point correspondences P = [(xr1; xt1); (xr2; xt2); ... (xrn; xtn)].
#           Glej bralnik



#a) HOMEOGRAPHY

def estimate_homography(pts1, pts2):
    """
    pts1: Nx2 array of points in first image (xr, yr)
    pts2: Nx2 array of points in second image (xt, yt)
    
    Returns 3x3 homography matrix H such that pts2 ~ H @ pts1
    """
    N = pts1.shape[0]
    A = []

    for i in range(N):
        xr, yr = pts1[i]
        xt, yt = pts2[i]

        # Equation 11 and 12
        A.append([xr, yr, 1, 0, 0, 0, -xt*xr, -xt*yr, -xt])
        A.append([0, 0, 0, xr, yr, 1, -yt*xr, -yt*yr, -yt])

    A = np.array(A)  # shape (2N, 9)

    # SVD
    U, S, VT = np.linalg.svd(A)
    h = VT[-1, :]  # last row of VT

    # Normalize so that h[8] == 1
    h = h / h[-1]

    # Reshape to 3x3
    H = h.reshape(3,3)
    return H

points = np.loadtxt("./data/newyork/newyork.txt")  # format: x1 y1 x2 y2
pts1 = points[:, :2]
pts2 = points[:, 2:]

img1 = cv2.imread("./data/newyork/newyork_a.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
img2 = cv2.imread("./data/newyork/newyork_b.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

plt.figure(figsize=(12,6))
plt.title("Newyork - homography")
a4.display_matches(img1, img2, pts1, pts2, [[0,0], [1,1], [2,2], [3,3]])

H = estimate_homography(pts1, pts2)

# Warp first image to second
warped = cv2.warpPerspective(src=img1, M=H, dsize=(img2.shape[1], img2.shape[0]))
print()
print(H)

plt.imshow(img2, cmap='gray')
plt.imshow(warped, cmap='jet', alpha=0.5)
plt.show()




points = np.loadtxt("./data/graf/graf.txt")  # format: x1 y1 x2 y2
pts1 = points[:, :2]
pts2 = points[:, 2:]

img1 = cv2.imread("./data/graf/graf_a.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
img2 = cv2.imread("./data/graf/graf_b.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

plt.figure(figsize=(12,6))
plt.title("Graf - homography")
a4.display_matches(img1, img2, pts1, pts2, [[0,0], [1,1], [2,2], [3,3]])

H = estimate_homography(pts1, pts2)

# Warp first image to second
warped = cv2.warpPerspective(src=img1, M=H, dsize=(img2.shape[1], img2.shape[0]))
print()
print(H)

plt.imshow(img2, cmap='gray')
plt.imshow(warped, cmap='jet', alpha=0.5)
plt.show()


#b) RANSAC
def get_line_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        k = np.inf
        n = x1  # vertical line, store x-intercept
    else:
        k = (y2 - y1) / (x2 - x1)
        n = y1 - k * x1
    return k, n

def point_line_distance(p, k, n):
    x, y = p
    if np.isinf(k):
        return abs(x - n)
    return abs(k*x - y + n) / np.sqrt(k**2 + 1)

def ransac_line_visual(points, max_iterations=20, distance_thresh=0.05):
    points_arr = np.array(points)
    best_inliers = []
    best_k, best_n = None, None
    
    fig, ax = plt.subplots()
    plt.ion()
    
    for it in range(max_iterations):
        ax.cla()
        random.shuffle(points)
        p1, p2 = points[:2]
        k, n = get_line_equation(p1, p2)
        
        # Compute distances to line
        distances = np.array([point_line_distance(p, k, n) for p in points])
        inliers = points_arr[distances < distance_thresh]
        outliers = points_arr[distances >= distance_thresh]
        
        # Update best line
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_k, best_n = k, n
        
        # Plot all points
        ax.plot(points_arr[:,0], points_arr[:,1], 'k.', label='points')
        if len(outliers) > 0:
            ax.plot(outliers[:,0], outliers[:,1], 'ro', label='outliers')
        if len(inliers) > 0:
            ax.plot(inliers[:,0], inliers[:,1], 'go', label='inliers')
        
        # Plot candidate line (red)
        x_vals = np.array([0, 1])
        if np.isinf(k):
            ax.axvline(n, color='r', linestyle='--', label='candidate line')
        else:
            ax.plot(x_vals, k*x_vals + n, 'r--', label='candidate line')
        
        # Plot best line so far (green)
        if best_k is not None:
            if np.isinf(best_k):
                ax.axvline(best_n, color='g', linestyle='-', label='best line')
            else:
                ax.plot(x_vals, best_k*x_vals + best_n, 'g-', label='best line')
        
        if len(best_inliers) > 0.95 * len(points):  # 95% of points are inliers
            break

        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        plt.axis('square')
        plt.legend()
        plt.draw()
        plt.pause(0.5)  # slow down iteration
        
    
    plt.ioff()
    plt.show()
    return best_k, best_n, best_inliers

np.random.seed(42)
N = 50
noise_scale = 0.1

# Generate points along a line with noise
x_vals = np.linspace(0, 1, N)
true_k, true_n = 0.5, 0.2
points = []
for x in x_vals:
    y = true_k*x + true_n + (np.random.random()-0.5)*noise_scale
    if 0 <= y <= 1:
        points.append((x,y))

k, n, inliers = ransac_line_visual(points, distance_thresh=0.05)
print()
print("Best line: y = {:.3f}x + {:.3f}, inliers: {}".format(k, n, len(inliers)))
print()


#c)
def reprojection_error(H, pts_src, pts_dst):
    """
    pts_src: Nx2 source points
    pts_dst: Nx2 destination points
    Returns: Nx1 Euclidean distances
    """
    N = pts_src.shape[0]
    pts_src_h = np.hstack([pts_src, np.ones((N,1))])  # make homogeneous  [x, y] -> [x, y, 1] (Stack arrays in sequence horizontally (column wise))
    pts_proj_h = (H @ pts_src_h.T).T                 # shape N x 3 
    # Multiply H (3x3) by pts_src_h.T (3xN) -> gives 3xN matrix.
    # .T -> Nx3.
    #  row is [x', y', z'] in homogeneous coordinates.)
    pts_proj = pts_proj_h[:, :2] / pts_proj_h[:, 2:3]  # normalize -> Divide x and y by z and go back to "D"
    errors = np.linalg.norm(pts_proj - pts_dst, axis=1)
    return errors


def ransac_homography(ptsA, ptsB, threshold=3.0, max_iters=1000, early_stop_ratio=0.8):
    """
    ptsA, ptsB: Nx2 matched points
    threshold: reprojection error threshold for inliers
    max_iters: maximum number of RANSAC iterations
    """
    N = ptsA.shape[0]
    best_H = None
    best_inliers = []
    num_iterations=0
    
    for i in range(max_iters):
        num_iterations += 1

        idx = np.random.choice(N, 4, replace=False)
        sample_A = ptsA[idx]
        sample_B = ptsB[idx]
        
        # Estimate homography from 4 points
        H = estimate_homography(sample_A, sample_B)
        
        errors = reprojection_error(H, ptsA, ptsB)
        
        inliers = np.where(errors < threshold)[0]
        
        # Update best solution
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            
            # early exit if enough inliers
            if len(best_inliers) > early_stop_ratio * N:
                break
    
    # Re-estimate using all inliers
    if len(best_inliers) > 4:
        best_H = estimate_homography(ptsA[best_inliers], ptsB[best_inliers])
    
    print("Number of iterations needed:", num_iterations)
    print()
    return best_H, best_inliers


A = cv2.imread("./data/newyork/newyork_a.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
B = cv2.imread("./data/newyork/newyork_b.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

matches, (XA,YA), (XB,YB) = find_matches(A, B)
ptsA = []
ptsB = []

ptsA = np.array([[XA[a], YA[a]] for a, b in matches])
ptsB = np.array([[XB[b], YB[b]] for a, b in matches])

best_H, best_inliers = ransac_homography(ptsA, ptsB)

print("Symmetric matches count after ransac_homography:", best_inliers.shape[0])

# Only keep matches that are inliers
inlier_matches = np.array([[i, i] for i in best_inliers])  # because ptsA[best_inliers] matches ptsB[best_inliers]

plt.figure(figsize=(12,6))
plt.title("MATCHES: Hessian keypoints + simple_descriptor + RANSAC + homography")
a4.display_matches(A, B, ptsA, ptsB, inlier_matches)


height, width = B.shape[:2]
warped_A = cv2.warpPerspective(A, best_H, (width, height))
plt.figure()
plt.imshow(warped_A, cmap='gray')
plt.title("Warped image A -> B")
plt.show()

plt.imshow(B, cmap='gray')
plt.imshow(warped_A, alpha=0.5)
plt.show()

#d)
# p = desired probability of finding at least one set of all-inlier points (e.g., 0.99)
# w = probability that a randomly selected point is an inlier (inlier ratio)
# n = minimal number of points needed to fit the model (for homography, n=4)

# I can roughly estimate w by looking at my matched points and counting how many appear roughly correct. For example, suppose I have 50 matched points, 
# and after running RANSAC I find 35 inliers, then: w = 32 / 41 = 0.78

# Typical value: p=0.99 (we want 99% chance of finding at least one all-inlier set).

# k = log(1 - p) / log(1 - w**n)
# Example: k = math.log(1 - 0.99) / math.log(1 - 0.78**4)
# k = log(0.01) / log(1 - 0.37)
# k = log(0.01) / log(0.63)
# k ~~ 9,7 iterations

def required_iterations(p, w, n):
    if w <= 0:
        return float('inf')
    denom = 1.0 - w**n
    if denom <= 0:
        return 1  # perfect inliers -> one sample is enough
    return math.log(1 - p) / math.log(denom)

def ransac_homography_adaptive(ptsA, ptsB, threshold=3.0, max_iters=1000,
                               desired_prob=0.99, early_stop_ratio=None, min_iters=10):
    """
    Adaptive RANSAC: recompute required iterations from best inlier ratio found so far.
    - desired_prob: target probability p (default 0.99)
    - early_stop_ratio: optional fraction of inliers to stop immediately (e.g. 0.8)
    - min_iters: minimum number of iterations to run (avoid premature stop)
    """
    N = ptsA.shape[0]
    best_H = None
    best_inliers = np.array([], dtype=int)
    iterations = 0
    n = 4  # minimal sample size for homography

    # initial conservative required_k
    current_w = 0.5  # conservative initial guess (or could be 0.1)
    required_k = math.ceil(required_iterations(desired_prob, current_w, n))

    while iterations < max_iters and iterations < required_k:
        iterations += 1

        # sample 4 random correspondences
        idx = np.random.choice(N, 4, replace=False)
        H = estimate_homography(ptsA[idx], ptsB[idx])

        errors = reprojection_error(H, ptsA, ptsB)
        inliers = np.where(errors < threshold)[0]

        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_H = H

            # recompute inlier ratio and required iterations
            current_w = best_inliers.size / N
            # avoid degenerate numeric cases
            if current_w <= 0:
                required_k = float('inf')
            else:
                req = required_iterations(desired_prob, current_w, n)
                required_k = math.ceil(req) if not math.isinf(req) else max_iters

            # optional early direct stop if fraction reached
            if early_stop_ratio is not None and best_inliers.size >= early_stop_ratio * N:
                print(f"Early stop by inlier fraction at iter={iterations}, inliers={best_inliers.size}/{N}")
                break

        # ensure we run at least min_iters
        if iterations >= min_iters and iterations >= required_k:
            break

    # re-fit H using all inliers for final estimate
    if best_inliers.size > 4:
        best_H = estimate_homography(ptsA[best_inliers], ptsB[best_inliers])

    print(f"RANSAC finished: iterations={iterations}, best_inliers={best_inliers.size}/{N}, required_k(now)={required_k}")
    return best_H, best_inliers

# The RANSAC iteration formula gives the number of iterations required so that,
# with probability p, at least one of the random minimal samples will contain
# only inliers — on average, assuming the inlier ratio w is known.
#
# Important caveats:
#
# 1. You must know w in advance.
#    The value you used (32/41) is the *final* observed inlier ratio, obtained
#    after RANSAC already succeeded. In practice, before running RANSAC you do
#    not know w, and early iterations have no guarantee that the empirical w
#    matches the final value.
#
# 2. Randomness and variance.
#    The formula gives a probabilistic guarantee. Even if k = 10 iterations are
#    theoretically enough, there is still a non-zero chance that none of the 10
#    random samples will be an all-inlier set.
#
# 3. Degenerate or bad minimal samples.
#    Some randomly chosen 4-point sets are degenerate (e.g., collinear or nearly
#    collinear). Such samples produce unstable or invalid homography estimates,
#    even if all 4 points are true inliers. This lowers the true success rate.
#
# 4. Model noise and threshold sensitivity.
#    If the inlier threshold is too strict or the data is noisy, the effective
#    inlier ratio w is smaller than what you estimate afterward. This increases
#    the required number of iterations.
#
# 5. Spatial correlation.
#    Inlier points may be spatially clustered. Random sampling assumes
#    independence, so the actual effectiveness can be lower than predicted.
#
# 6. Structured outliers.
#    If outliers are not random (e.g., many false matches are consistent),
#    RANSAC can be misled. These structured outliers reduce the true w and
#    increase the required number of iterations.


#e)
def warp_perspective_manual(img, H): 
    h, w = img.shape[:2]
    #Colour image
    #h, w, c = img.shape
    H_inv = np.linalg.inv(H)
    out = np.zeros_like(img)

    # Precompute pixel coordinates -> Generate a grid of all target pixels
    ys, xs = np.indices((h, w))  # shape (h, w)

    # Flatten and convert to homogeneous
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(h*w)], axis=1).T   # shape 3 x N
    # pts = [x0, x1, ..., xN;
    #        y0, y1, ..., yN;
    #        1,  1,  ...,  1 ]


    # Backward mapping: target -> source
    src_pts_h = H_inv @ pts

    # Normalize
    src_x = src_pts_h[0] / src_pts_h[2]
    src_y = src_pts_h[1] / src_pts_h[2]

    src_x_round = np.round(src_x).astype(int)
    src_y_round = np.round(src_y).astype(int)

    # Create mask for valid pixel positions
    mask = (
        (src_x_round >= 0) & (src_x_round < w) &
        (src_y_round >= 0) & (src_y_round < h)
    )

    # Fill output using only valid mapped pixels
    out_flat = out.ravel()
    img_flat = img.ravel()

    # Colour version
    # Flattened indexing
    # out_flat = out.reshape(-1, 3)    # (N, 3)
    # img_flat = img.reshape(-1, 3)    # (N, 3)

    # mapping: each pixel (x,y) in output takes value from corresponding source
    src_idx = src_y_round * w + src_x_round
    out_flat[mask] = img_flat[src_idx[mask]]

    return out


# Forward mapping (source -> destination):
# - We take each pixel in the source image and push it forward using H.
# - Problem: many destination pixels never get assigned a value -> holes.
# - Because some source pixels may map to the same destination pixel,
#   while many destination pixels receive nothing.
# - This is forbidden by the assignment.

# Backward mapping (destination -> source):
# - For each output pixel (x', y'), we apply H^-1 to find where it came from:
#       (x, y) = H^-1 * (x', y', 1)
# - Then we copy the value from the source pixel at (x, y).
# - This guarantees that every output pixel gets exactly one source pixel
#   (or zero if the mapped coordinates fall outside the source image).
# - Result: no holes appear in the output image.

points = np.loadtxt("./data/newyork/newyork.txt")  # format: x1 y1 x2 y2
pts1 = points[:, :2]
pts2 = points[:, 2:]

img1 = cv2.imread("./data/newyork/newyork_a.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
img2 = cv2.imread("./data/newyork/newyork_b.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

plt.figure(figsize=(12,6))
plt.title("Newyork - homography - manual warp perspective")
a4.display_matches(img1, img2, pts1, pts2, [[0,0], [1,1], [2,2], [3,3]])

H = estimate_homography(pts1, pts2)
warped = warp_perspective_manual(img1, H)

plt.imshow(img2, cmap='gray')
plt.imshow(warped, cmap='jet', alpha=0.5)
plt.show()