import a5_utils as a5
import cv2, random, math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.image as mpimg
import os

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 1 
#a) Blizje kot je predmet, manjsi je pz in posledicno vecji je d

#b)
def disparity(f, T, pz):
    return (T * f) / pz

f = 2.5
T = 120
pz = np.arange(1, 11)
d = disparity(f, T, pz)
print("Disparity v mm:", d)
plt.plot(pz, d, marker='o')
plt.xlabel("p_z (mm)")        # X-axis label
plt.ylabel("Disparity (pixels)")  # Y-axis label
plt.title("Disparity vs Depth")   # Plot title
plt.grid(True)                     # Optional grid
plt.show()


#c)
f = 2.5
T = 120
pixelWidth = 7.4e-3
x1 = 550
x2 = 300
#x2 = 540
pz = (T * f) / ((x1 - x2)*pixelWidth)
print("Razdalja v mm:\n", pz)

#d)
def ncc(A, B):
    A = A.astype(float)
    B = B.astype(float)

    A_mean = A.mean()
    B_mean = B.mean()

    up = np.sum((A - A_mean) * (B - B_mean))
    down = np.sqrt(np.sum((A - A_mean)**2) * np.sum((B - B_mean)**2))

    if down == 0:
        return -1  # invalid match

    return up / down

def compute_disparity(left, right, patch_size=10, max_disp=80):
    h, w = left.shape
    half = patch_size // 2
    
    disparity = np.zeros((h, w), dtype=float)

    for y in range(half, h-half):
        for x in range(half, w-half):

            patch_L = left[y-half:y+half+1, x-half:x+half+1]

            best_ncc = -1
            best_dx = 0

            # search only to the left
            for d in range(max_disp):
                xr = x - d
                if xr < half:
                    break

                patch_R = right[y-half:y+half+1, xr-half:xr+half+1]

                score = ncc(patch_L, patch_R)

                if score > best_ncc:
                    best_ncc = score
                    best_dx = d

            disparity[y, x] = best_dx

    return disparity


left = cv2.imread("./data/disparity/office_left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("./data/disparity/office_right.png", cv2.IMREAD_GRAYSCALE)

left = cv2.resize(left, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
right = cv2.resize(right, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

# File paths
l2r_file = "disp_L2R.npy"
r2l_file = "disp_R2L.npy"

if os.path.exists(l2r_file) and os.path.exists(r2l_file):
    disp_L2R = np.load(l2r_file)
    disp_R2L = np.load(r2l_file)
    print("Disparities loaded successfully.")
else:
    disp_L2R = compute_disparity(left, right, patch_size=10, max_disp=60)
    disp_R2L = compute_disparity(right, left, patch_size=10, max_disp=60)

    np.save(l2r_file, disp_L2R)
    np.save(r2l_file, disp_R2L)
    print("Disparities computed successfully.")

plt.imshow(disp_L2R, cmap='jet')
plt.colorbar()
plt.show()


#e)
# Convert R2L to L2R coordinates:
disp_R2L_shifted = np.zeros_like(disp_L2R)
h, w = disp_L2R.shape

for y in range(h):
    for x in range(w):
        d = int(disp_L2R[y, x])
        xr = int(x - d)
        if 0 <= xr < w:
            disp_R2L_shifted[y, x] = disp_R2L[y, xr]


consistent = np.abs(disp_L2R - disp_R2L_shifted) < 1
disp_merged = np.where(consistent, disp_L2R, 0)
disp_filtered = cv2.medianBlur(disp_merged.astype(np.uint8), 5)


h, w = left.shape
reconstructed = np.zeros_like(left)

# Verification
for y in range(h):
    for x in range(w):
        d = int(disp_filtered[y, x])
        xr = int(x - d)
        if 0 <= xr < w:
            reconstructed[y, xr] = left[y, x]


# offset + compare disparity

plt.figure(figsize=(12,4))
plt.subplot(131)
plt.title("Right image (50%)")
plt.imshow(right, cmap='gray')
plt.axis('off')

plt.subplot(132)
plt.title("Filtered disparity")
plt.imshow(disp_filtered, cmap='jet')
plt.colorbar()
plt.axis('off')

plt.subplot(133)
plt.title("Reconstructed image")
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()



#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 2
#a) Glej bralnik -> TO-DO
# Epipolar line for (column=0, row=2):
#       l' = (0, 1, -1).T -> v = 1
# 
# Epipolar line for (column=1, row=0):
#       l' = (1, 0, -1).T -> u = 1

# Epipole in the right camera:
#       Lies at infinity (no finite epipole)

#b)
def fundamental_matrix(Xl, Xr):
    # Convert inputs to float
    Xl = Xl.astype(float)
    Xr = Xr.astype(float)
    
    n = Xl.shape[0]
    if n < 8:
        raise ValueError("Need at least 8 points")

    Xl_norm, T1 = a5.normalize_points(Xl)
    Xr_norm, T2 = a5.normalize_points(Xr)

    A = []
    for i in range(n):
        u, v = Xl_norm[i, 0], Xl_norm[i, 1]
        up, vp = Xr_norm[i, 0], Xr_norm[i, 1]
        A.append([u*up, v*up, up, u*vp, v*vp, vp, u, v, 1])
    A = np.array(A)

    U, S, VT = np.linalg.svd(A)
    f = VT[-1]
    F = f.reshape(3, 3)

    U2, S2, VT2 = np.linalg.svd(F)
    S2[2] = 0  # force rank 2
    F_rank2 = U2 @ np.diag(S2) @ VT2

    F_final = T2.T @ F_rank2 @ T1

    U3, S3, VT3 = np.linalg.svd(F_final)
    e_left = VT3[-1]
    e_right = U3[:, -1]

    e_left /= e_left[2]
    e_right /= e_right[2]

    return F_final, e_left, e_right

pts7 = np.loadtxt("./data/house/2D/house.007.corners")
pts8 = np.loadtxt("./data/house/2D/house.008.corners")
matches = a5.read_matches("./data/house/2D/house.nview-corners", 7, 8)

Xl = []
Xr = []

for idx7, idx8 in matches:
    if idx7 >= 0 and idx8 >= 0:
        Xl.append(pts7[idx7])
        Xr.append(pts8[idx8])

Xl = np.array(Xl)
Xr = np.array(Xr)


F, e1, e2 = fundamental_matrix(Xl, Xr)
print("Fundamental matrix:\n", F)
print("Left epipole:", e1)
print("Right epipole:", e2)



img7 = cv2.imread("./data/house/images/house.007.png")
img8 = cv2.imread("./data/house/images/house.008.png")
h7, w7 = img7.shape[:2]
h8, w8 = img8.shape[:2]


# Show results
plt.figure(figsize=(10,6))
plt.imshow(img8[:,:,::-1])
plt.title("Epipolar lines in image 8")

for i in range(len(Xl)):
    p1 = np.array([Xl[i,0], Xl[i,1], 1])
    line = F @ p1
    a5.draw_epiline(line, h8, w8, clr='cyan')
    # also plot the corresponding point
    plt.scatter(Xr[i,0], Xr[i,1], c='yellow', s=10)

plt.show()

plt.figure(figsize=(10,6))
plt.imshow(img7[:,:,::-1])
plt.title("Epipolar lines in image 7")

for i in range(len(Xl)):
    p2 = np.array([Xr[i,0], Xr[i,1], 1])
    line = F.T @ p2
    a5.draw_epiline(line, h7, w7, clr='magenta')
    plt.scatter(Xl[i,0], Xl[i,1], c='yellow', s=10)

plt.show()

F_ref = np.loadtxt("./data/house/F_7_8.txt")
error = np.linalg.norm(F / F[2,2] - F_ref / F_ref[2,2])
print("Difference from reference:", error)




#c)
def point_to_line_distance(point, line):
    # point: [x, y]
    a, b, c = line
    x, y = point[:2]  # use first two coordinates
    return abs(a*x + b*y + c) / np.sqrt(a*a + b*b)

# (a) ----------------------------------------------------------


p1 = np.array([160, 463, 1])  # left image
p2 = np.array([128, 437, 1])  # right image

# epipolar line in right image:
l_r = F @ p1

error = point_to_line_distance(p2, l_r)

print("Reprojection error:", error)

# (b) ----------------------------------------------------------
errors = []

for p1, p2 in zip(Xl, Xr):
    p1_h = np.array([p1[0], p1[1], 1])
    p2_h = np.array([p2[0], p2[1], 1])

    # Epipolar lines
    l_r = F @ p1_h
    l_l = F.T @ p2_h

    d1 = point_to_line_distance(p2, l_r)   # p2 -> line from p1
    d2 = point_to_line_distance(p1, l_l)   # p1 -> line from p2

    errors.append(0.5 * (d1 + d2))

avg_error = np.mean(errors)

print("Average symmetric reprojection error:", avg_error)




#d)
img1 = cv2.imread("./data/desk/DSC02638.JPG", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./data/desk/DSC02639.JPG", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract point coordinates for matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
# mask is 1 for inliers, 0 for outliers
pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

print("Estimated Fundamental Matrix:\n", F)
print(f"Number of inliers: {len(pts1_inliers)}/{len(pts1)}")

plt.figure(figsize=(12,6))
img_matches = cv2.drawMatches(
    img1, kp1, img2, kp2, 
    [matches[i] for i in range(len(matches)) if mask[i]], 
    None, matchColor=(0,255,0), singlePointColor=(255,0,0)
)
plt.imshow(img_matches[:,:,::-1])
plt.title("Inlier Matches after RANSAC")
plt.axis("off")
plt.show()

def draw_epilines(img1, img2, pts1, pts2, F):
    """Draw epilines in img1 corresponding to pts2"""
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    for r, pt1, pt2 in zip(lines1, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [img1.shape[1], -(r[2]+r[0]*img1.shape[1])/r[1]])
        img1_color = cv2.line(img1_color, (x0,y0), (x1,y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(pt1.astype(int)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(pt2.astype(int)), 5, color, -1)
    
    return img1_color, img2_color

img1_epi, img2_epi = draw_epilines(img1, img2, pts1_inliers, pts2_inliers, F)

plt.figure(figsize=(12,6))
plt.subplot(121); plt.imshow(img1_epi[:,:,::-1]); plt.title("Epipolar Lines on Image 1"); plt.axis('off')
plt.subplot(122); plt.imshow(img2_epi[:,:,::-1]); plt.title("Epipolar Lines on Image 2"); plt.axis('off')
plt.show()








#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 3
#a)
def triangulate(Xl, Xr, P1, P2):
    """
    Xl, Xr : Nx2 image points
    P1, P2 : 3x4 projection matrices
    returns: Nx3 array of 3D points
    """

    assert Xl.shape == Xr.shape
    n = Xl.shape[0]

    X3D = []

    for i in range(n):
        x, y = Xl[i]
        xp, yp = Xr[i]

        # Build matrix A
        A = np.array([
            x * P1[2] - P1[0],
            y * P1[2] - P1[1],
            xp * P2[2] - P2[0],
            yp * P2[2] - P2[1]
        ])

        # Solve AX = 0
        _, _, VT = np.linalg.svd(A)
        X = VT[-1]
        X = X / X[3]          # homogeneous -> Euclidean

        X3D.append(X[:3])

    return np.array(X3D)




pts7 = np.loadtxt("./data/house/2D/house.007.corners")
pts8 = np.loadtxt("./data/house/2D/house.008.corners")

matches = a5.read_matches("./data/house/2D/house.nview-corners", 7, 8)

Xl, Xr = [], []
for i7, i8 in matches:
    Xl.append(pts7[i7])
    Xr.append(pts8[i8])

Xl = np.array(Xl)
Xr = np.array(Xr)

P1 = np.loadtxt("./data/house/3D/house.007.P")
P2 = np.loadtxt("./data/house/3D/house.008.P")

X3D = triangulate(Xl, Xr, P1, P2)

T = np.array([
    [-1, 0, 0],
    [ 0, 0,-1],
    [ 0, 1, 0]
])

X3D_t = (T @ X3D.T).T



img7 = mpimg.imread("./data/house/images/house.007.png")
img8 = mpimg.imread("./data/house/images/house.008.png")

N = X3D_t.shape[0]

colors = cm.tab20(np.linspace(0, 1, N))

fig = plt.figure(figsize=(18, 6))
fig.suptitle("Triangulation: Image 7 and Image 8", fontsize=18, y=0.98)

# Image 7
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img7, cmap="gray")
ax1.set_title("Image 7")
ax1.axis("off")

for i, (x, y) in enumerate(Xl):
    ax1.scatter(x, y, color=colors[i], s=20)
    ax1.text(x, y, str(i), color=colors[i], fontsize=8)

# Image 8
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(img8, cmap="gray")
ax2.set_title("Image 8")
ax2.axis("off")

for i, (x, y) in enumerate(Xr):
    ax2.scatter(x, y, color=colors[i], s=20)
    ax2.text(x, y, str(i), color=colors[i], fontsize=8)

# 3D view
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.set_title("Triangulated 3D points")

for i, p in enumerate(X3D_t):
    ax3.scatter(p[0], p[1], p[2], color=colors[i], s=30)
    ax3.text(p[0], p[1], p[2], str(i), fontsize=8)

ax3.set_xlabel("X")
ax3.set_ylabel("Y")
ax3.set_zlabel("Z")

plt.tight_layout()
plt.show()





#b)
# Load images
img1 = cv2.imread("./data/house/images/house.001.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("./data/house/images/house.004.png", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

print(f"Number of matches: {len(matches)}")


# OpenCV RANSAC fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)

pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

print(f"Number of inliers: {len(pts1_inliers)}")

P1 = np.loadtxt("./data/house/3D/house.001.P")
P2 = np.loadtxt("./data/house/3D/house.004.P")

X3D = triangulate(pts1_inliers, pts2_inliers, P1, P2)

T = np.array([[-1, 0, 0],
              [0, 0, -1],
              [0, 1, 0]])

X3D_t = (T @ X3D.T).T





N = X3D_t.shape[0]
colors = cm.tab20(np.linspace(0, 1, N))

fig = plt.figure(figsize=(18, 8))
fig.suptitle("SIFT + RANSAC + triangulation: Image 1 and Image 4", fontsize=18, y=0.98)

# Image 1
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(img1, cmap='gray')
ax1.set_title("Image 1 keypoints")
ax1.axis("off")
for i, (x,y) in enumerate(pts1_inliers):
    ax1.scatter(x, y, color=colors[i])
    ax1.text(x, y, str(i), color=colors[i], fontsize=8)

# Image 4
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(img2, cmap='gray')
ax2.set_title("Image 4 keypoints")
ax2.axis("off")
for i, (x,y) in enumerate(pts2_inliers):
    ax2.scatter(x, y, color=colors[i])
    ax2.text(x, y, str(i), color=colors[i], fontsize=8)

# 3D view
ax3 = fig.add_subplot(1,3,3, projection='3d')
ax3.set_title("Triangulated 3D points")
for i, p in enumerate(X3D_t):
    ax3.scatter(p[0], p[1], p[2], color=colors[i])
    ax3.text(p[0], p[1], p[2], str(i), fontsize=8)

ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
plt.tight_layout()
plt.show()
