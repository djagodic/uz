import a6_utils as a6
import cv2, random, math
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 1
#a) Glej bralnik
def directPCA(X):
    X = np.array(X).T
    m, N = X.shape
    mu = np.mean(X, axis=1, keepdims=True)
    Xd = X - mu
    #print("Xd", Xd)
    C = (1/(N-1))*Xd@Xd.T
    # print("C", C)
    U, S, _ = np.linalg.svd(C)
    # print("U", U)
    # print("S - eigen values", S)
    return mu, C, U, S

vhod = np.loadtxt("./data/points1a.txt")
# 3 4
# 3 6
# 7 6
# 6 4
mu, C, eigvecs, eigvals = directPCA(vhod)

print("C", C)
print("U", eigvecs)
print("S - eigen values", eigvals)

plt.figure()

# Plot data points
plt.scatter(vhod[:, 0], vhod[:, 1], label="Data points")

# Add point indices
for i, (x, y) in enumerate(vhod):
    plt.text(x + 0.05, y + 0.05, str(i + 1), fontsize=9)

# Plot mean value μ
plt.scatter(mu[0,0], mu[1,0], color='black', marker='x', s=100, label="Mean μ")

# Length scaling for visibility
scale = np.sqrt(eigvals)

# First eigenvector (red line)
plt.plot(
    [mu[0], mu[0] + scale[0]*eigvecs[0,0]],
    [mu[1], mu[1] + scale[0]*eigvecs[1,0]],
    color='red',
    linewidth=2,
    label="1st eigenvector"
)

# Second eigenvector (green line)
plt.plot(
    [mu[0], mu[0] + scale[1]*eigvecs[0,1]],
    [mu[1], mu[1] + scale[1]*eigvecs[1,1]],
    color='green',
    linewidth=2,
    label="2nd eigenvector"
)

# Gaussian ellipse
a6.drawEllipse(mu.flatten(), C)

plt.axis('equal')
plt.legend()
plt.title("PCA with Eigenvectors, Mean and Point Indices")
plt.show()



#b) + c)
vhod = np.loadtxt("./data/points.txt")
# 1 0
# 6 2
# 5 4
# 1 3
# 0 1
mu, C, U, S = directPCA(vhod)
eigvecs = U
eigvals = S

plt.figure()

# Plot data points
plt.scatter(vhod[:, 0], vhod[:, 1], label="Data points")

# Add point indices
for i, (x, y) in enumerate(vhod):
    plt.text(x + 0.05, y + 0.05, str(i + 1), fontsize=9)

# Plot mean value μ
plt.scatter(mu[0,0], mu[1,0], color='black', marker='x', s=100, label="Mean μ")

# Length scaling for visibility
scale = np.sqrt(eigvals)

# First eigenvector (red line)
plt.plot(
    [mu[0], mu[0] + scale[0]*eigvecs[0,0]],
    [mu[1], mu[1] + scale[0]*eigvecs[1,0]],
    color='red',
    linewidth=2,
    label="1st eigenvector"
)

# Second eigenvector (green line)
plt.plot(
    [mu[0], mu[0] + scale[1]*eigvecs[0,1]],
    [mu[1], mu[1] + scale[1]*eigvecs[1,1]],
    color='green',
    linewidth=2,
    label="2nd eigenvector"
)

# Gaussian ellipse
a6.drawEllipse(mu.flatten(), C)

plt.axis('equal')
plt.legend()
plt.title("PCA with Eigenvectors, Mean and Point Indices")
plt.show()


#Question: What do you notice about the relationship between the eigenvectors and the data? What happens to the eigenvectors if you change the data or add more points?
#    The eigenvectors represent the principal directions of variance in the data.
#       - The first eigenvector (red) is aligned with the direction in which the data varies the most.
#       - The second eigenvector (green) is orthogonal to the first and represents the direction of the second-largest variance.

#    The eigenvectors depend directly on the covariance matrix, which is computed from the data. Therefore:
#       - If the data distribution changes, the eigenvectors rotate to match the new principal directions.
#       - Adding more points that follow the same trend usually stabilizes the eigenvectors.
#       - Adding outliers or points in a different direction can significantly change the orientation and magnitude of the eigenvectors.
#       - If the data becomes more isotropic (similar variance in all directions), the eigenvalues become similar and the eigenvectors become less informative.


#d)
cum_var = np.cumsum(eigvals)

# Normalize so the largest cumulative value is 1
cum_var_norm = cum_var / cum_var[-1]

variance_retained = eigvals[0] / np.sum(eigvals)
print(f"\nVariance retained using only first eigenvector: {variance_retained * 100:.2f}%")


# Plot
plt.figure()
plt.plot(
    range(1, len(cum_var_norm) + 1),
    cum_var_norm,
    marker='o'
)
plt.xticks(range(1, len(cum_var_norm) + 1))
plt.xlabel("Number of eigenvectors")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative Normalized Eigenvalues")
plt.grid(True)
plt.show()

#e)
# Original data (N x 2)
X = vhod

# Center data
X_centered = X - mu.T

# Project into PCA space (Y = Uᵀ (x - μ))
Y = (U.T @ X_centered.T).T   # shape: (N, 2)

# Remove lowest variance component (second eigenvector)
Y_reduced = Y.copy()
Y_reduced[:, 1] = 0

# Reconstruct back to original space (x̂ = U y + μ)
X_reconstructed = (U @ Y_reduced.T).T + mu.T


plt.figure()

# Original points
plt.scatter(
    X[:, 0], X[:, 1],
    color='blue', label='Original points'
)

# Reconstructed points
plt.scatter(
    X_reconstructed[:, 0], X_reconstructed[:, 1],
    color='red', marker='x', label='Reconstructed points'
)

# Draw projection lines
for i in range(len(X)):
    plt.plot(
        [X[i,0], X_reconstructed[i,0]],
        [X[i,1], X_reconstructed[i,1]],
        color='gray', linestyle='--', linewidth=0.8
    )

# First eigenvector line
scale = S[0]
plt.plot(
    [mu[0] - scale*U[0,0], mu[0] + scale*U[0,0]],
    [mu[1] - scale*U[1,0], mu[1] + scale*U[1,0]],
    color='green', linewidth=2, label='1st eigenvector'
)

# Mean
plt.scatter(mu[0], mu[1], color='black', marker='x', s=100, label='Mean μ')

plt.axis('equal')
plt.legend()
plt.title("PCA Projection onto First Eigenvector")
plt.show()


#Question: What happens to the reconstructed points? Where is the data projected to?
#   What happens to the reconstructed points?
#       After removing the second PCA component and reconstructing the data:
#       - The reconstructed points no longer coincide with the original points.
#       - Each point is shifted toward the direction of the first eigenvector.
#       - The variation in the direction of the second eigenvector is completely removed.
#       - The distance between an original point and its reconstruction corresponds to the reconstruction error, which equals the variance associated with the discarded eigenvector.

#    Where is the data projected to?
#       All reconstructed points lie on a straight line defined by:
#       - the first eigenvector
#       - passing through the mean of the data
#   This line represents the 1-dimensional PCA subspace that captures the maximum possible variance of the original data.


#f)

# Original input points
X = vhod  # shape (N, 2)
qpoint = np.array([6, 6])  # query point

# --- Step 1: Find closest point in original space ---
distances = np.linalg.norm(X - qpoint, axis=1)
closest_index = np.argmin(distances)
print(f"\nClosest point to qpoint in original space: {X[closest_index]} (index {closest_index})")

# --- Step 2: Center the original points ---
X_centered = X - mu.T  # shape (N,2)

# --- Step 3: Project points into PCA space ---
Y = (U.T @ X_centered.T).T  # shape (N,2) lahko bi dal kar X_centered @ U in se znebim nekaj transponiranj (zunaj dodan .T, da je tudi Y oblike (N,2))
q_centered = qpoint - mu.flatten()
q_Y = U.T @ q_centered       # shape (2,)

# --- Step 4: Remove second component (lowest variance) ---
Y_reduced = Y.copy()
Y_reduced[:, 1] = 0

q_Y_reduced = q_Y.copy()
q_Y_reduced[1] = 0

# --- Step 5: Reconstruct back to original space ---
X_reconstructed = (U @ Y_reduced.T).T + mu.T   # shape (N,2)
q_reconstructed = U @ q_Y_reduced + mu.flatten()  # shape (2,)

# --- Step 6: Find closest point after PCA projection ---
distances_reduced = np.linalg.norm(X_reconstructed - q_reconstructed, axis=1)
closest_index_reduced = np.argmin(distances_reduced)
print(f"Closest point to qpoint after PCA projection: {X[closest_index_reduced]} (index {closest_index_reduced})")

# --- Step 7: Visualization ---
plt.figure()

# Original points
plt.scatter(X[:,0], X[:,1], color='blue', label='Original points')

# Reconstructed points
plt.scatter(X_reconstructed[:,0], X_reconstructed[:,1], color='red', marker='x', label='Projected points')

# Query point and its projection
plt.scatter(qpoint[0], qpoint[1], color='green', marker='o', s=100, label='qpoint')
plt.scatter(q_reconstructed[0], q_reconstructed[1], color='orange', marker='x', s=100, label='qpoint projected')

# Lines connecting original -> projected points
for i in range(len(X)):
    plt.plot(
        [X[i,0], X_reconstructed[i,0]],
        [X[i,1], X_reconstructed[i,1]],
        color='gray', linestyle='--', linewidth=0.8
    )

# Draw first eigenvector for reference
scale = S[0]
plt.plot(
    [mu[0,0] - scale*U[0,0], mu[0,0] + scale*U[0,0]],
    [mu[1,0] - scale*U[1,0], mu[1,0] + scale*U[1,0]],
    color='green', linewidth=2, label='1st eigenvector'
)

# Plot mean
plt.scatter(mu[0,0], mu[1,0], color='black', marker='x', s=100, label='Mean μ')

plt.axis('equal')
plt.legend()
plt.title("PCA Projection and Nearest Point")
plt.show()


#g)
X = np.loadtxt("./data/points_50D.txt")   # shape (N,50)
mu, C, U, S = directPCA(X)
eigvals = S

# Cumulative normalized eigenvalue (variance explained)
cum_var = np.cumsum(eigvals)
cum_var_norm = cum_var / cum_var[-1]   # normalize to 1

# Determine how many eigenvectors to keep for 80% variance
num_to_keep = np.searchsorted(cum_var_norm, 0.8) + 1  # +1 because searchsorted returns index
print(f"Number of eigenvectors to KEEP for >=80% variance: {num_to_keep}")

# Optional: How many eigenvectors can be set to zero
num_to_zero = X.shape[1] - num_to_keep
print(f"Number of eigenvectors that can be set to zero: {num_to_zero}")


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 2
#a)
def dualPCA(X):
    X = np.array(X).T
    m, N = X.shape
    mu = np.mean(X, axis=1, keepdims=True)
    Xd = X - mu
    C = (1/(N-1))*Xd.T@Xd
    U, S, _ = np.linalg.svd(C)

    U = Xd @ U @ np.diag(np.sqrt(1 / ((S + 1e-15) * (N - 1))))
    return mu, C, U, S

vhod = np.loadtxt("./data/points.txt")
# 3 4
# 3 6
# 7 6
# 6 4
mu, C, U, S = dualPCA(vhod)

print("\nC", C)
print("U", U)
print("S - eigen values", S)

#b)
X = vhod
# Center data
X_centered = X - mu.T

# Project into PCA space (Y = Uᵀ (x - μ))
Y = (U.T @ X_centered.T).T   # shape: (N, 2)

# Reconstruct back to original space (x̂ = U y + μ)
X_reconstructed = (U @ Y.T).T + mu.T

print("Original:\n", X)
print("Reconstructed:\n", X_reconstructed)

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 3
#a)

def load_image_series_to_matrix(folder_path):
    """
    Reads all images from folder_path, converts to grayscale,
    reshapes them into column vectors, and stacks them into a matrix.

    Returns:
        X : np.ndarray of shape (m*n, N)
        image_shape : (m, n)
    """

    files = sorted(os.listdir(folder_path))
    vectors = []

    for fname in files:
        path = os.path.join(folder_path, fname)

        # Load image and convert to grayscale
        img = Image.open(path).convert("L")   # 'L' = grayscale
        img_arr = np.asarray(img, dtype=np.float64)

        # Save shape from first image
        if not vectors:
            image_shape = img_arr.shape

        # Flatten to column vector
        vec = img_arr.reshape(-1, 1)
        vectors.append(vec)

    # Stack column-wise → (m*n, 64)
    X = np.hstack(vectors)

    return X, image_shape


X, img_shape = load_image_series_to_matrix("./data/faces/1")

print("\nData matrix shape:", X.shape)      # (m*n, 64)
print("Original image shape:", img_shape)

#b)
mu, C, U, S = dualPCA(X.T)

plt.figure(figsize=(12,4))

for i in range(5):
    eigenface = U[:, i].reshape(img_shape)

    plt.subplot(1, 5, i + 1)
    plt.imshow(eigenface, cmap="gray")
    plt.axis("off")
    plt.title(f"Eigenface {i+1}")

plt.suptitle("First 5 Eigenfaces")
plt.show()
    
# Take first image
x0 = X[:, 0:1]  # (D, 1)

# Center
x0c = x0 - mu

# Project to PCA space
y0 = U.T @ x0c   # (N, 1)

# Reconstruct back
x0_reconstructed = U @ y0 + mu

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(x0.reshape(img_shape), cmap="gray")
plt.title("Original image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(x0_reconstructed.reshape(img_shape), cmap="gray")
plt.title("Reconstructed image")
plt.axis("off")

plt.show()

# Numerically 
#   - Each eigenface is an eigenvector of the covariance matrix
#   - It represents a principal direction of variance
#   - Pixel values indicate how strongly each pixel contributes to that variation
# In the context of faces
#   - Eigenfaces encode:
#   - Head orientation
#   - Lighting direction
#   - Face proportions
#   - Expression patterns
# They are not faces, but basis patterns
# Any face ≈ linear combination of eigenfaces + mean face


#c)
# Take first image
x = X[:, 0:1].copy()   # shape (D,1)
original = x.copy()

# Choose a pixel index (e.g. center pixel)
pixel_idx1 = X.shape[0] // 2
x[pixel_idx1 + 41, 0] = 0    # set ONE pixel to zero

# Project to PCA space
y = U.T @ (x - mu)

# Reconstruct
x_rec = U @ y + mu

# Visualize
plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.imshow(original.reshape(img_shape), cmap="gray")
plt.title("Original image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(x.reshape(img_shape), cmap="gray")
plt.title("Image with 1 pixel set to zero")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(x_rec.reshape(img_shape), cmap="gray")
plt.title("After PCA projection + reconstruction")
plt.axis("off")

plt.show()

# Numerically
#   Only one pixel was changed initially
#   After PCA reconstruction:
#       - Many pixels are modified
#       - The change spreads across the entire image
# Why?
#   Because PCA reconstruction uses global basis vectors (eigenfaces).
#   Each eigenface affects all pixels.


# Original centered image
x = X[:, 0:1]
x_c = x - mu

# Project to PCA space
y = U.T @ x_c

print("\nFirst 5 PCA coefficients before:", y[:5].ravel())
# Zero one of the first 5 PCA components
component_idx = 1   # e.g. first eigenvector
y_mod = y.copy()
y_mod[component_idx, 0] = 0
print("After zeroing:", y_mod[:5].ravel())

# Reconstruct
x_rec_pca = U @ y_mod + mu

# Visualize
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(x.reshape(img_shape), cmap="gray")
plt.title("Original image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(x_rec_pca.reshape(img_shape), cmap="gray")
plt.title("After zeroing PCA component")
plt.axis("off")

plt.show()

# Visually
#   The face changes globally
#   Typical effects:
#       - Lighting change
#       - Loss of expression
#       - Blurring of structure
#       - The image still looks like a face
# Numerically
#   All pixels are affected
#   But in a structured and meaningful way

# When a single pixel is modified in image space, only one dimension of the data is altered.
# However, after projection to PCA space and reconstruction, the modification spreads across almost all pixels because PCA eigenvectors are global basis functions.

# In contrast, modifying a single PCA coefficient affects all pixels immediately, but in a structured manner corresponding to a principal mode of variation in the data.
# The resulting image remains plausible and face-like.

# Therefore, PCA space provides a compact, meaningful representation where small changes correspond to large but semantically meaningful changes in the original image.

#d)
# --- Assume X, U, mu, img_shape are already defined from dualPCA ---
# X: (D, N)  -> D = m*n pixels, N = number of images
# U: (D, N)  -> PCA eigenvectors
# mu: (D,1)  -> mean image
# img_shape: (m, n)

# Pick a random image
# np.random.seed(42)
# idx = np.random.randint(0, X.shape[1])
# print(f"\nTaken image 00{idx + 1}.png for task d")
# x = X[:, idx:idx+1]          # shape (D,1)
x = X[:, 0:1]

# Project to PCA space
y = U.T @ (x - mu)           # shape (N,1)

# Number of components to keep
components_list = [32, 16, 8, 4, 2, 1]

# Prepare figure
plt.figure(figsize=(15, 4))

for i, k in enumerate(components_list):
    y_mod = np.zeros_like(y)
    y_mod[:k, 0] = y[:k, 0]   # retain only first k components
    
    # Reconstruct
    x_rec = U @ y_mod + mu
    
    # Plot
    plt.subplot(1, len(components_list), i+1)
    plt.imshow(x_rec.reshape(img_shape), cmap='gray')
    plt.title(f'{k} components')
    plt.axis('off')

plt.suptitle("Effect of number of PCA components on reconstruction")
plt.show()



#e)
X, img_shape = load_image_series_to_matrix("./data/faces/2")

print("\nData matrix shape:", X.shape)      # (m*n, 64)
print("Original image shape:", img_shape)

mu, C, U, S = dualPCA(X.T)

# --- Compute average image ---
x_avg = np.mean(X, axis=1, keepdims=True)  # shape (mn,1)

# Project average image to PCA space
y_avg = U.T @ (x_avg - mu)

# Select a more important eigenvector (e.g., 2nd component)
eig_idx = 3  # 0 = first, 1 = second

# Animation parameters
frames = 100
scale = 1000
t = np.linspace(0, 2*np.pi, frames)  # 1 period

# plt.ion()  # turn on interactive mode
# fig, ax = plt.subplots()
# ax.imshow(x_avg.reshape(img_shape), cmap='gray')
# ax.axis('off')
# plt.title(f"Manipulating eigenvector {eig_idx+1}")

# for s in np.sin(t) * scale:
#     y_mod = y_avg.copy()
#     y_mod[eig_idx, 0] = s
#     x_rec = U @ y_mod + mu
#     ax.imshow(x_rec.reshape(img_shape), cmap='gray')
#     plt.pause(0.05)
#     #print(s)

# plt.ioff()  # turn off interactive mode
# plt.show()

plt.ion()  # turn on interactive mode
fig, ax = plt.subplots()
im = ax.imshow(x_avg.reshape(img_shape), cmap='gray', vmin=x_avg.min(), vmax=x_avg.max())
ax.axis('off')
plt.title(f"Manipulating eigenvector {eig_idx+1}")

for s in np.sin(t) * scale:
    y_mod = y_avg.copy()
    y_mod[eig_idx, 0] = s
    x_rec = U @ y_mod + mu
    im.set_data(x_rec.reshape(img_shape))
    plt.pause(0.05)

plt.ioff()  # turn off interactive mode
plt.show()


# Select two eigenvectors (e.g., 1st and 2nd)
scale = 1000
eig_x = 0
eig_y = 1

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(x_avg.reshape(img_shape), cmap='gray', vmin=x_avg.min(), vmax=x_avg.max())
ax.axis('off')
plt.title("Manipulating two eigenvectors")

for angle in np.linspace(0, 2*np.pi, frames):
    y_mod = y_avg.copy()
    y_mod[eig_x, 0] = scale * np.cos(angle)
    y_mod[eig_y, 0] = scale * np.sin(angle)
    x_rec = U @ y_mod + mu
    im.set_data(x_rec.reshape(img_shape))
    plt.pause(0.05)

plt.ioff()
plt.show()


#f)
X, img_shape = load_image_series_to_matrix("./data/faces/1")
mu, C, U, S = dualPCA(X.T)

elephant = cv2.imread("./data/elephant.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64)
x_elephant = elephant.reshape(-1, 1)

# Center the image using mean of faces
x_centered = x_elephant - mu

# Project to PCA space
y_elephant = U.T @ x_centered

# Reconstruct back to image space
x_reconstructed = U @ y_elephant + mu

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(x_elephant.reshape(img_shape), cmap='gray')
plt.title("Original elephant image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(x_reconstructed.reshape(img_shape), cmap='gray')
plt.title("Reconstructed from face PCA")
plt.axis('off')

plt.show()

# Original elephant image: clearly visible, natural image.
# Reconstructed image:
#   - Likely looks very distorted or face-like
#   - PCA cannot represent features outside the variance of the training set (faces)
#   - Only the “closest face-like projection” is kept
# Comment for report:
#   PCA is a linear projection based on the variance in the training data. When a foreign image is projected into a PCA space built from faces, 
#   the reconstructed image tries to express the foreign object using the face eigenvectors. As a result, the elephant is poorly reconstructed and 
#   may resemble a blurry or face-like version of itself, demonstrating that PCA is limited to the data distribution it was trained on.


#g)
folder_path = "./data/my_faces"
img_shape = [64, 64]
# Load images
files = sorted(os.listdir(folder_path))
X = []
for f in files:
    path = os.path.join(folder_path, f)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if not X:
            img_shape = img.shape

    X.append(img.reshape(-1))  # flatten to vector

#X = np.array(X).T  # D x N, each column is an image -> v dual PCA naredim to

# Compute PCA
mu, _, U, _ = dualPCA(X)  # D x N

def recognize_face(face_img, mu, U, threshold=3000):
    x = face_img.reshape(-1,1)
    y = U.T @ (x - mu)
    x_rec = U @ y + mu
    error = np.linalg.norm(x - x_rec)
    recognized = error < threshold
    return recognized, x_rec, error


# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, img_shape)
        recognized, x_rec, error = recognize_face(face_resized, mu, U, threshold=6000)

        color = (0,255,0) if recognized else (0,0,255)
        label = "ME" if recognized else "Unknown"

        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Recognition PCA", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
