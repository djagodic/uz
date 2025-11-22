import a4_utils as a4
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ---------- Helpers: derivatives & Hessian ----------
def hessian_matrix(image, sigma):
    """Return (Ixx, Iyy, Ixy) computed with correct separable convolutions."""
    G = a4.gauss(sigma)
    D = a4.gaussdx(sigma)

    # First derivatives (separable)
    Ix = a4.convolve(a4.convolve(image, G.T), D)   # smooth vertically (G.T) then differentiate horizontally (D)
    Iy = a4.convolve(a4.convolve(image, G),   D.T) # smooth horizontally (G) then differentiate vertically (D.T)

    # Second derivatives
    Ixx = a4.convolve(a4.convolve(Ix, G.T), D)
    Iyy = a4.convolve(a4.convolve(Iy, G),   D.T)

    # Mixed derivative: derivative of Ix w.r.t. y OR derivative of Iy w.r.t. x
    # Using Ix smoothed in y and differentiated in y.T
    Ixy = a4.convolve(a4.convolve(Ix, G), D.T)

    return Ixx, Iyy, Ixy

def hessian_points(image, sigma=1.2, thresh=0.04):
    Ixx, Iyy, Ixy = hessian_matrix(image, sigma)

    det = Ixx * Iyy - Ixy * Ixy

    # normalize determinant to [0,1] (assignment asks for normalization)
    det_norm = (det - det.min()) / (det.max() - det.min() + 1e-12)

    # threshold after normalization
    det_norm = np.where(det_norm > thresh, det_norm, 0.0)

    return det_norm

# ---------- Non-maximum suppression ----------
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

# ---------- Hellinger and correspondence ----------
def hellinger(a, b):
    return np.sqrt(0.5 * np.sum((np.sqrt(a) - np.sqrt(b))**2))

def find_correspondences(desc1, desc2):
    """For each descriptor in desc1 find index of closest descriptor in desc2 using Hellinger."""
    n1 = desc1.shape[0]
    n2 = desc2.shape[0]
    # vectorized distance computation: compute sqrt then pairwise L2 square via broadcasting
    # desc are histograms (non-negative)
    A = np.sqrt(desc1)           # shape (n1, D)
    B = np.sqrt(desc2)           # shape (n2, D)
    # compute pairwise squared distances: |A_i - B_j|^2 = |A_i|^2 + |B_j|^2 - 2 A_i·B_j
    A2 = np.sum(A**2, axis=1)[:, None]     # (n1,1)
    B2 = np.sum(B**2, axis=1)[None, :]     # (1,n2)
    AB = A.dot(B.T)                        # (n1,n2)
    D2 = A2 + B2 - 2 * AB                  # squared L2 distances on sqrt-histograms
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(0.5 * D2)                  # Hellinger distance
    # best matches (indices in desc2)
    best_idx = np.argmin(D, axis=1)
    # return list-of-pairs
    return [[i, int(best_idx[i])] for i in range(n1)]

# ---------- Symmetric matching ----------
def find_matches(image_A, image_B,
                 sigma=1.2, thresh_det=0.04, nms_box=7, unique_max=False,
                 desc_radius=25, desc_bins=16):
    # detect keypoints
    detA = hessian_points(image_A, sigma=sigma, thresh=thresh_det)
    detB = hessian_points(image_B, sigma=sigma, thresh=thresh_det)

    ptsA = non_maxima_suppression(detA, box_size=nms_box, thresh=0.0, unique_max=unique_max)
    ptsB = non_maxima_suppression(detB, box_size=nms_box, thresh=0.0, unique_max=unique_max)

    YA, XA = np.nonzero(ptsA)
    YB, XB = np.nonzero(ptsB)
    print("Detected keypoints: A =", len(XA), " B =", len(XB))

    # descriptors (simple_descriptors expects (I, Y, X))
    descA = a4.simple_descriptors(image_A, YA, XA, n_bins=desc_bins, radius=desc_radius, sigma=1.0)
    descB = a4.simple_descriptors(image_B, YB, XB, n_bins=desc_bins, radius=desc_radius, sigma=1.0)
    print("Descriptors shapes:", descA.shape, descB.shape)

    # find best matches both directions
    lr = find_correspondences(descA, descB)  # list of [i, j]
    rl = find_correspondences(descB, descA)  # list of [j, i] from B->A

    # map arrays
    lr_map = np.array([b for _, b in lr], dtype=int)   # for each i (A) -> best j (B)
    rl_map = np.array([a for _, a in rl], dtype=int)   # for each j (B) -> best i (A)

    # symmetric matches: indices i such that rl_map[ lr_map[i] ] == i
    valid = np.where((lr_map < len(rl_map)) & (rl_map[lr_map] == np.arange(len(lr_map))))[0]
    matches = np.stack([valid, lr_map[valid]], axis=1) if valid.size else np.zeros((0,2), dtype=int)

    # convert to coordinates for display
    ptsA_vis = np.array([[XA[i], YA[i]] for i, _ in matches])
    ptsB_vis = np.array([[XB[j], YB[j]] for _, j in matches])

    print("Symmetric matches count:", matches.shape[0])
    return matches, (XA, YA), (XB, YB), ptsA_vis, ptsB_vis, detA, detB

# ---------- Demo run with recommended parameters ----------
A = cv2.imread("./data/graf/graf_a_small.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
B = cv2.imread("./data/graf/graf_b_small.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)/255

matches, (XA,YA), (XB,YB), ptsA_vis, ptsB_vis, detA, detB = find_matches(
    A, B,
    sigma=1.2,          # smaller sigma for small images
    thresh_det=0.04,    # normalized det threshold (try 0.02..0.08)
    nms_box=5,          # NMS window
    unique_max=False,   # allow ties (less strict)
    desc_radius=25,     # descriptor radius (try 20..30)
    desc_bins=16
)

# debug: show normalized determinants
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.imshow(detA, cmap='gray'); plt.title('det A norm'); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(detB, cmap='gray'); plt.title('det B norm'); plt.axis('off')
plt.show()

# visualize matches (if any)
if ptsA_vis.shape[0] > 0:
    a4.display_matches(A, ptsA_vis, B, ptsB_vis)
else:
    print("No symmetric matches to display. Try lowering thresh_det or increasing desc_radius.")
