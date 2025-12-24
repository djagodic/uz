import a6_utils as a6
import cv2, random, math
import numpy as np
from matplotlib import pyplot as plt

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
    # print("S - singular values -> sqrt for eigvals", S)
    return mu, C, U, S

vhod = np.loadtxt("./data/points1a.txt")
# 3 4
# 3 6
# 7 6
# 6 4
mu, C, eigvecs, eigvals = directPCA(vhod)

print("C", C)
print("U", eigvecs)
print("S - singular values -> sqrt for eigvals", eigvals)
print("eigen values", eigvals)

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
print(f"Variance retained using only first eigenvector: {variance_retained * 100:.2f}%")


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
print(f"Closest point to qpoint in original space: {X[closest_index]} (index {closest_index})")

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

# --- Step 5: Cumulative normalized eigenvalue (variance explained) ---
cum_var = np.cumsum(eigvals)
cum_var_norm = cum_var / cum_var[-1]   # normalize to 1

# --- Step 6: Determine how many eigenvectors to keep for 80% variance ---
num_to_keep = np.searchsorted(cum_var_norm, 0.8) + 1  # +1 because searchsorted returns index
print(f"Number of eigenvectors to KEEP for >=80% variance: {num_to_keep}")

# --- Optional: How many eigenvectors can be set to zero ---
num_to_zero = X.shape[1] - num_to_keep
print(f"Number of eigenvectors that can be set to zero: {num_to_zero}")
