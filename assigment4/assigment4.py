import a4_utils as a4
import cv2
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


def hessian_points(image, sigma, thresh):
    I_xx, I_yy = second_partial_derivatives(image, sigma)
    I_xy, I_yx = second_partial_mixed_derivatives(image, sigma)

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

def non_maxima_suppression3(det, box_size=3, thresh=0.3):
    det = np.where(det > thresh, det, 0)

    #non-maxima suppresion
    suppressed = np.zeros_like(det, dtype=np.float64)

    # paddanje
    radius = box_size // 2
    padded = np.pad(det, radius, mode='constant', constant_values=0)

    ys, xs = np.nonzero(det)
    for y, x in zip(ys, xs):
        val = det[y, x]

        # Extract neighborhood
        window = padded[y : y + box_size, x : x + box_size]

        # Check if center is the maximum
        #if val == np.max(window):
        #ohrani le ce je edini maximum
        max_val = np.max(window)
        if val == max_val and np.sum(window == max_val) == 1:
            suppressed[y, x] = val
        
    return suppressed

def non_maxima_suppression2(det, box_size=3, thresh=0.0):
    det = np.where(det > thresh, det, 0)

    suppressed = np.zeros_like(det)
    R = box_size // 2
    padded = np.pad(det, R)

    ys, xs = np.nonzero(det)

    for y,x in zip(ys,xs):
        window = padded[y:y+box_size, x:x+box_size]
        if det[y,x] == np.max(window):
            suppressed[y,x] = det[y,x]

    return suppressed

def non_maxima_suppression(det, box_size=7, thresh=0.0, unique_max=False):
    """Keep pixels equal to the local maximum in box_size window (centered)."""
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

points3_harris = hessian_points(graf, sigma=3, thresh=0.2)
points3 = non_maxima_suppression(points3_harris, box_size=7, thresh=0.5)
points7_harris = hessian_points(graf, sigma=7, thresh=0.2)
points7 = non_maxima_suppression(points7_harris, box_size=5, thresh=0.5)
points13_harris = hessian_points(graf, sigma=13, thresh=0.2)
points13 = non_maxima_suppression(points13_harris, box_size=5, thresh=0.4)


plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(points3_harris, cmap="gray")
plt.title("σ=3")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(points7_harris, cmap="gray")
plt.title("σ=7")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(points13_harris, cmap="gray")
plt.title("σ=13")
plt.axis('off')
plt.subplot(2, 3, 4)
plot_points(graf_color, points3)
plt.axis('off')
plt.subplot(2, 3, 5)
plot_points(graf_color, points7)
plt.axis('off')
plt.subplot(2, 3, 6)
plot_points(graf_color, points13)
plt.axis('off')

plt.show()

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

points3_harris = harris(graf, sigma=3, thresh=0.2)
points3 = non_maxima_suppression(points3_harris, box_size=7, thresh=0.6)
points7_harris = harris(graf, sigma=7, thresh=0.2)
points7 = non_maxima_suppression(points7_harris, box_size=7, thresh=0.5)
points13_harris = harris(graf, sigma=13, thresh=0.2)
points13 = non_maxima_suppression(points13_harris, box_size=7, thresh=0.59)


plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(points3_harris, cmap="gray")
plt.title("σ=3")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(points7_harris, cmap="gray")
plt.title("σ=7")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(points13_harris, cmap="gray")
plt.title("σ=13")
plt.axis('off')
plt.subplot(2, 3, 4)
plot_points(graf_color, points3)
plt.axis('off')
plt.subplot(2, 3, 5)
plot_points(graf_color, points7)
plt.axis('off')
plt.subplot(2, 3, 6)
plot_points(graf_color, points13)
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

# Detect points (you can reuse your Hessian/Harris)
detA = hessian_points(A, sigma=3, thresh=0.2)
ptsA = non_maxima_suppression(detA, box_size=7, thresh=0.5)
detB = hessian_points(B, sigma=3, thresh=0.2)
ptsB = non_maxima_suppression(detB, box_size=7, thresh=0.5)

YA, XA = np.nonzero(ptsA)
YB, XB = np.nonzero(ptsB)
print("Task a) Detected keypoints: A =", len(XA), " B =", len(XB))

# Compute descriptors
descA = a4.simple_descriptors(A, YA, XA)
descB = a4.simple_descriptors(B, YB, XB)

# Match
corr = find_correspondences(descA, descB)

# Convert correspondences to coordinate pairs
ptsA_vis = []
ptsB_vis = []

for a, b in corr:
    ptsA_vis.append([XA[a], YA[a]])
    ptsB_vis.append([XB[b], YB[b]])

ptsA_vis = np.array(ptsA_vis)
ptsB_vis = np.array(ptsB_vis)

# Display result
plt.figure(figsize=(12,5))
a4.display_matches(A, ptsA_vis, B, ptsB_vis)

#b)
def find_matches2(image_A, image_B):
    detA = hessian_points(image_A, sigma=1.2, thresh=0.04)
    ptsA = non_maxima_suppression(detA, box_size=7, thresh=0.3)

    detB = hessian_points(image_B, sigma=1.2, thresh=0.04)
    ptsB = non_maxima_suppression(detB, box_size=7, thresh=0.3)

    YA, XA = np.nonzero(ptsA)
    YB, XB = np.nonzero(ptsB)

    print("Points A:", len(XA))
    print("Points B:", len(XB))


    descA = a4.simple_descriptors(image_A, YA, XA, radius=25, sigma=1.0)
    descB = a4.simple_descriptors(image_B, YB, XB, radius=25, sigma=1.0)

    lr = find_correspondences(descA, descB)  # left to right
    rl = find_correspondences(descB, descA)  # right to left

    lr_map = np.array([b for _, b in lr])   # length = N_A
    rl_map = np.array([a for a, _ in rl])   # length = N_B

    # # === 4. Select symmetric matches ===
    # matches = []
    # for i in range(len(lr_map)):
    #     j = lr_map[i]      # best match A_i → B_j
    #     if j < len(rl_map) and rl_map[j] == i:
    #         matches.append([i, j])


    # indices where rl_map[lr_map[i]] == i
    valid = np.where((lr_map < len(rl_map)) & (rl_map[lr_map] == np.arange(len(lr_map))))[0]

    matches = np.stack([valid, lr_map[valid]], axis=1)


    return matches, (XA, YA), (XB, YB)


def find_matches(image_A, image_B, sigma=3, thresh_det=0.1, nms_box=7, unique_max=True, desc_radius=35, desc_bins=16):
    # detect keypoints
    detA = hessian_points(image_A, sigma=sigma, thresh=thresh_det)
    detB = hessian_points(image_B, sigma=sigma, thresh=thresh_det)

    ptsA = non_maxima_suppression(detA, box_size=nms_box, thresh=0.5, unique_max=unique_max)
    ptsB = non_maxima_suppression(detB, box_size=nms_box, thresh=0.5, unique_max=unique_max)


    YA, XA = np.nonzero(ptsA)
    YB, XB = np.nonzero(ptsB)
    print("Detected keypoints: A =", len(XA), " B =", len(XB))

    # descriptors (simple_descriptors expects (I, Y, X))
    descA = a4.simple_descriptors(image_A, YA, XA, n_bins=desc_bins, radius=desc_radius, sigma=2)
    descB = a4.simple_descriptors(image_B, YB, XB, n_bins=desc_bins, radius=desc_radius, sigma=2)
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

ptsA_vis = []
ptsB_vis = []

for i, j in matches:
    ptsA_vis.append([XA[i], YA[i]])
    ptsB_vis.append([XB[j], YB[j]])

ptsA_vis = np.array(ptsA_vis)
ptsB_vis = np.array(ptsB_vis)

plt.figure(figsize=(12,6))
a4.display_matches(A, ptsA_vis, B, ptsB_vis)


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 3