import a5_utils as a5
import cv2, random, math
import numpy as np
from matplotlib import pyplot as plt

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


#c)
f = 2.5
T = 120
pixelWidth = 7.4e-3
x1 = 550
#x2 = 300
x2 = 540
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

# disp_L2R = compute_disparity(left, right, patch_size=10, max_disp=80)

# plt.imshow(disp_L2R, cmap='jet')
# plt.colorbar()
# plt.show()


# #disp_L2R = compute_disparity(left, right, patch_size=10)
# disp_R2L = compute_disparity(right, left, patch_size=10)

# #e)
# # Convert R2L to L2R coordinates:
# disp_R2L_shifted = np.zeros_like(disp_L2R)
# h, w = disp_L2R.shape

# for y in range(h):
#     for x in range(w):
#         d = disp_L2R[y, x]
#         xr = x - d
#         if xr >= 0:
#             disp_R2L_shifted[y, x] = disp_R2L[y, xr]

# consistent = np.abs(disp_L2R - disp_R2L_shifted) < 1
# disp_merged = np.where(consistent, disp_L2R, 0)
# disp_filtered = cv2.medianBlur(disp_merged.astype(np.uint8), 5)


# h, w = left.shape
# reconstructed = np.zeros_like(left)

# # Verification by image warping
# for y in range(h):
#     for x in range(w):
#         d = disp_filtered[y, x]
#         xr = x - int(d)
#         if 0 <= xr < w:
#             reconstructed[y, xr] = left[y, x]


# plt.figure(figsize=(12,4))
# plt.subplot(131); plt.title("Right image"); plt.imshow(right, cmap='gray')
# plt.subplot(132); plt.title("Reconstructed"); plt.imshow(reconstructed, cmap='gray')
# plt.subplot(133); plt.title("Difference"); plt.imshow(abs(reconstructed - right), cmap='gray')
# plt.show()


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 2
#a) Glej bralnik -> TO-DO

#b)
def fundamental_matrix(Xl, Xr):
    # Convert inputs to float
    Xl = Xl.astype(float)
    Xr = Xr.astype(float)
    
    n = Xl.shape[0]
    if n < 8:
        raise ValueError("Need at least 8 points")

    # --- 1. Normalize points ---
    Xl_norm, T1 = a5.normalize_points(Xl)
    Xr_norm, T2 = a5.normalize_points(Xr)

    # --- 2. Build matrix A ---
    A = []
    for i in range(n):
        u, v = Xl_norm[i, 0], Xl_norm[i, 1]
        up, vp = Xr_norm[i, 0], Xr_norm[i, 1]
        A.append([u*up, v*up, up, u*vp, v*vp, vp, u, v, 1])
    A = np.array(A)

    # --- 3. Solve Af = 0 using SVD ---
    U, S, VT = np.linalg.svd(A)
    f = VT[-1]
    F = f.reshape(3, 3)

    # --- 4. Enforce rank 2 on F ---
    U2, S2, VT2 = np.linalg.svd(F)
    S2[2] = 0  # force rank 2
    F_rank2 = U2 @ np.diag(S2) @ VT2

    # --- 5. Denormalize ---
    F_final = T2.T @ F_rank2 @ T1

    # --- 6. Compute epipoles ---
    U3, S3, VT3 = np.linalg.svd(F_final)
    e_left = VT3[-1]          # right nullspace of F = last row of V^T
    e_right = U3[:, -1]       # left nullspace = last column of U

    e_left /= e_left[2]
    e_right /= e_right[2]

    return F_final, e_left, e_right


#epipolar line for point
# l_right = F @ x
# l_left = F.T @ x_right

pts7 = np.loadtxt("./data/house/2D/house.007.corners")
pts8 = np.loadtxt("./data/house/2D/house.008.corners")
matches = a5.read_matches("./data/house/2D/house.nview-corners", 7, 8)

Xl = []
Xr = []

for idx7, idx8 in matches:
    if idx7 >= 0 and idx8 >= 0:  # valid match
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




# Create a figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Image 8 with epilines ---
axes[0].imshow(img8[:, :, ::-1])
axes[0].set_title("Epipolar lines in image 8")
for i in range(len(Xl)):
    p1 = np.array([Xl[i, 0], Xl[i, 1], 1])
    line = F @ p1
    a5.draw_epiline(line, h8, w8, clr='cyan')
    axes[0].scatter(Xr[i, 0], Xr[i, 1], c='yellow', s=10)


# --- Image 7 with epilines ---
axes[1].imshow(img7[:, :, ::-1])
axes[1].set_title("Epipolar lines in image 7")
for i in range(len(Xl)):
    p2 = np.array([Xr[i, 0], Xr[i, 1], 1])
    line = F.T @ p2
    a5.draw_epiline(line, h7, w7, clr='magenta')
    axes[1].scatter(Xl[i, 0], Xl[i, 1], c='yellow', s=10)

# Remove axes ticks for a cleaner look
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()






F_ref = np.loadtxt("./data/house/F_7_8.txt")
error = np.linalg.norm(F / F[2,2] - F_ref / F_ref[2,2])
print("Difference from reference:", error)




#e)
def point_to_line_distance(point, line):
    u, v = point
    a, b, c = line
    return abs(a*u + b*v + c) / np.sqrt(a*a + b*b)

# (a) ----------------------------------------------------------

F = np.loadtxt("./data/house/F_7_8.txt")  # Load the reference fundamental matrix

p1 = np.array([160, 463, 1])  # left image
p2 = np.array([128, 437])     # right image

# epipolar line in right image:
l_r = F @ p1

error = point_to_line_distance(p2, l_r)

print("Reprojection error:", error)

# (b) ----------------------------------------------------------

# Load keypoints
pts7 = np.loadtxt("./data/house/2D/house.007.corners")
pts8 = np.loadtxt("./data/house/2D/house.008.corners")

# Load matches using provided helper
matches = a5.read_matches("./data/house/2D/house.nview-corners", 7, 8)

# Extract matching pairs
Xl = []
Xr = []
for m in matches:
    if m[0] >= 0 and m[1] >= 0:
        Xl.append(pts7[m[0]])
        Xr.append(pts8[m[1]])

Xl = np.array(Xl)
Xr = np.array(Xr)

# Compute F using your function
F, e1, e2 = fundamental_matrix(Xl, Xr)

# Compute average symmetric reprojection error
errors = []

for p1, p2 in zip(Xl, Xr):
    p1_h = np.array([p1[0], p1[1], 1])
    p2_h = np.array([p2[0], p2[1], 1])

    # Epipolar lines
    l_r = F @ p1_h
    l_l = F.T @ p2_h

    d1 = point_to_line_distance(p2, l_r)   # p2 → line from p1
    d2 = point_to_line_distance(p1, l_l)   # p1 → line from p2

    errors.append(0.5 * (d1 + d2))

avg_error = np.mean(errors)

print("Average symmetric reprojection error:", avg_error)

