import a3_utils as a3
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import os


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 1 
#a)
#Glej bralnik

#b)
def gaussdx(sigma):

    limit = 2*(int(3*sigma + 1)) + 1

    def g(x):
        return -(1 / ((sigma**3) * np.sqrt(2 * np.pi))) * x * np.exp(-(x**2) / (2 * sigma**2))
    
    kernel = np.zeros(limit, dtype=np.float64)
    x = np.arange(-int(limit/2), int(limit/2)+1)
    kernel = g(x)
    kernel /= np.sum(np.abs(kernel))
    return kernel

kernel1 = gaussdx(0.5)
kernel2 = gaussdx(1)
kernel3 = gaussdx(2)
kernel4 = gaussdx(3)
kernel5 = gaussdx(4)

plt.figure(figsize=(15, 8))
plt.title("Gaussian Derivative x Kernel")
plt.plot(np.arange(-2, 3), kernel1, color="b", label="sigma = 0.5")
plt.plot(np.arange(-4, 5), kernel2, color="orange", label="sigma = 1")
plt.plot(np.arange(-7, 8), kernel3, color="g", label="sigma = 2")
plt.plot(np.arange(-10, 11), kernel4, color="r", label="sigma = 3")
plt.plot(np.arange(-13, 14), kernel5, color="purple", label="sigma = 4")

plt.legend()
plt.show()

#c)
def gauss(sigma):
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    
    limit = 2*(int(3*sigma + 1)) + 1

    def g(x):
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x**2) / (2 * sigma**2))
    
    kernel = np.zeros(limit, dtype=np.float64)
    x = np.arange(-int(limit/2), int(limit/2)+1)
    kernel = g(x)
    kernel /= kernel.sum()
    return kernel


sigma = 5
#cv2.filter2D dela corelacijo in ne konvolucije -> ročno flipnemo filter
G = np.fliplr(np.expand_dims(gauss(sigma), axis=0))
D = np.fliplr(np.expand_dims(gaussdx(sigma), axis=0))

impulse = np.zeros((50, 50))
impulse[25, 25] = 1

convolve_a = cv2.filter2D(impulse, ddepth=-1, kernel=G)
convolve_a2 = cv2.filter2D(convolve_a, ddepth=-1, kernel=G.T)

convolve_b = cv2.filter2D(impulse, ddepth=-1, kernel=G)
convolve_b2 = cv2.filter2D(convolve_b, ddepth=-1, kernel=D.T)

convolve_c = cv2.filter2D(impulse, ddepth=-1, kernel=D)
convolve_c2 = cv2.filter2D(convolve_c, ddepth=-1, kernel=G.T)

convolve_d = cv2.filter2D(impulse, ddepth=-1, kernel=G.T)
convolve_d2 = cv2.filter2D(convolve_d, ddepth=-1, kernel=D)

convolve_e = cv2.filter2D(impulse, ddepth=-1, kernel=D.T)
convolve_e2 = cv2.filter2D(convolve_e, ddepth=-1, kernel=G)

plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1)
plt.imshow(impulse, cmap="gray")
plt.title("I")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(convolve_a2, cmap="gray")
plt.title("(I * G) * G$^{T}$")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(convolve_b2, cmap="gray")
plt.title("(I * G) * D$^{T}$")
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(convolve_c2, cmap="gray")
plt.title("(I * D) * G$^{T}$")
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(convolve_d2, cmap="gray")
plt.title("(I * G$^{T}$) * D")
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(convolve_e2, cmap="gray")
plt.title("(I * D$^{T}$) * G")
plt.axis('off')

plt.show()

#d)
def partial_derivatives(image, sigma):
    G = np.fliplr(np.expand_dims(gauss(sigma), axis=0))
    D = np.fliplr(np.expand_dims(gaussdx(sigma), axis=0))

    image_x1 = cv2.filter2D(image, ddepth=-1, kernel=G.T)
    image_x = cv2.filter2D(image_x1, ddepth=-1, kernel=D)

    image_y1 = cv2.filter2D(image, ddepth=-1, kernel=G)
    image_y = cv2.filter2D(image_y1, ddepth=-1, kernel=D.T)

    return image_x, image_y

def second_partial_derivatives(image, sigma):
    image_x, image_y = partial_derivatives(image, sigma)

    G = np.fliplr(np.expand_dims(gauss(sigma), axis=0))
    D = np.fliplr(np.expand_dims(gaussdx(sigma), axis=0))

    image_xx1 = cv2.filter2D(image_x, ddepth=-1, kernel=G.T)
    image_xx = cv2.filter2D(image_xx1, ddepth=-1, kernel=D)

    image_yy1 = cv2.filter2D(image_y, ddepth=-1, kernel=G)
    image_yy = cv2.filter2D(image_yy1, ddepth=-1, kernel=D.T)

    return image_xx, image_yy

def second_partial_mixed_derivatives(image, sigma):
    image_x, image_y = partial_derivatives(image, sigma)

    G = np.fliplr(np.expand_dims(gauss(sigma), axis=0))
    D = np.fliplr(np.expand_dims(gaussdx(sigma), axis=0))

    image_xy1 = cv2.filter2D(image_x, ddepth=-1, kernel=G)
    image_xy = cv2.filter2D(image_xy1, ddepth=-1, kernel=D.T)

    image_yx1 = cv2.filter2D(image_y, ddepth=-1, kernel=G.T)
    image_yx = cv2.filter2D(image_yx1, ddepth=-1, kernel=D)

    return image_xy, image_yx

def gradient_magnitude(image, sigma):
    """
    firstly returns magnitude, second angle
    """
    image_x, image_y = partial_derivatives(image, sigma)
    m = np.sqrt(image_x**2 + image_y**2)
    m_norm = m / np.max(m)
    fi = np.arctan2(image_y, image_x)

    return m_norm, fi


sigma = 2
image = cv2.imread("./images/museum.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255

image_x, image_y = partial_derivatives(image, sigma)
image_xx, image_yy = second_partial_derivatives(image, sigma)
image_xy, image_yx = second_partial_mixed_derivatives(image, sigma)
image_m, image_fi = gradient_magnitude(image, sigma)


m_norm = image_m
# Normalize angles fi (-pi, pi) -> (0. 2*pi) -> (0, 1)
h = (image_fi % (2*np.pi)) / (2*np.pi)

# Full saturation
s = np.ones_like(h)
# Compose HSV image (H, S, V)
hsv = np.stack((h, s, m_norm), axis=-1)
# Convert HSV -> RGB for Matplotlib display
rgb = colors.hsv_to_rgb(hsv)

plt.figure(figsize=(12, 9))
plt.subplot(3, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis('off')
plt.subplot(3, 3, 2)
plt.imshow(image_x, cmap="gray")
plt.title("I$_{x}$")
plt.axis('off')
plt.subplot(3, 3, 3)
plt.imshow(image_y, cmap="gray")
plt.title("I$_{y}$")
plt.axis('off')
plt.subplot(3, 3, 4)
plt.imshow(image_xx, cmap="gray")
plt.title("I$_{xx}$")
plt.axis('off')
plt.subplot(3, 3, 5)
plt.imshow(image_xy, cmap="gray")
plt.title("I$_{xy}$")
plt.axis('off')
plt.subplot(3, 3, 6)
plt.imshow(image_yy, cmap="gray")
plt.title("I$_{yy}$")
plt.axis('off')
plt.subplot(3, 3, 7)
plt.imshow(image_m, cmap="gray")
plt.title("I$_{mag}$")
plt.axis('off')
plt.subplot(3, 3, 8)
plt.imshow(image_fi, cmap="gray")
plt.title("I$_{dir}$")
plt.axis('off')
plt.subplot(3, 3, 9)
plt.imshow(rgb)
plt.title("I$_{dir}$ (HSV)")
plt.axis('off')

plt.show()

#e)
def gradient_histogram_feature(image, sigma=1, grid_size=8, n_angle_bins=8):
    H, W = image.shape
    m, fi = gradient_magnitude(image, sigma)

    # gradient -> (0, 2*pi)
    fi = (fi + 2*np.pi) % (2*np.pi)

    cell_h = H // grid_size
    cell_w = W // grid_size
    # ys = np.linspace(0, H, grid_size+1, dtype=int)
    # xs = np.linspace(0, W, grid_size+1, dtype=int)
    feature = []

    bin_width = 2*np.pi / n_angle_bins

    for i in range(grid_size):
        for j in range(grid_size):

            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            # y0, y1 = ys[i], ys[i+1]
            # x0, x1 = xs[j], xs[j+1]
            
            cell_m = m[y0:y1, x0:x1].ravel()
            cell_fi = fi[y0:y1, x0:x1].ravel()

            # koti
            angle_bins = (cell_fi // bin_width).astype(int)
            angle_bins = np.clip(angle_bins, 0, n_angle_bins-1)

            # utežen histogram
            hist = np.zeros(n_angle_bins)
            for k in range(cell_m.size):
                hist[angle_bins[k]] += cell_m[k]

            # s = hist.sum()
            # if s > 1e-8:
            #     hist = hist / s
            
            feature.extend(hist)


    norm = np.sum(np.abs(feature))
    feature = feature / norm

    return np.array(feature)


def compute_save_gradient_features(image_dir, sigma=1, grid_size=8, n_angle_bins=8, save_path="grad_features.npy"):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    features = []

    for f in image_files:
        img_path = os.path.join(image_dir, f)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)/255
        feat = gradient_histogram_feature(img_gray, sigma, grid_size, n_angle_bins)
        features.append(feat)
    
    features = np.array(features)
    np.save(save_path, features)
    print(f"Saved {len(features)} gradient features to {save_path}")
    return image_files, features

def compare_histograms(hist1, hist2, distance: str = "l2"):
    eps = 1e-10
    match distance:
        case "l2":
            #občutljiva na dominantne barve
            return np.sqrt(np.sum((hist1 - hist2)**2))
        case "chi2":
            #rabi epsilon, če je bin slučajno 0, zmanjša dominanco velikih binov
            return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + eps))
        case "intersection":
            #ni prava metrika
            return 1.0 - np.sum(np.minimum(hist1, hist2)) 
        case "hellinger":
            #dobra za primerjavo normaliziranih bravnih histogramov, lahko preceni male bine
            return np.sqrt(0.5 * np.sum((np.sqrt(hist1) - np.sqrt(hist2))**2))

def retrieve_similar_images(query_idx, image_files, histograms, distance="l2", top_k=5):

    query_hist = histograms[query_idx]
    
    # razdalja do vseh
    dists = [compare_histograms(query_hist, hist, distance) for hist in histograms]

    # Sortiraj indexe po dolzini (mala = bolj podobno) -> dobimo indexe ki bodo sortirali tabelo
    sorted_indices = np.argsort(dists)

    # zbriši iskano
    top_indices = [i for i in sorted_indices if i != query_idx][:top_k]

    plt.figure(figsize=(18,8))
    plt.suptitle(f"Distance measure: {distance}")
    
    query_img = cv2.cvtColor(cv2.imread(os.path.join("./dataset", image_files[query_idx])), cv2.COLOR_BGR2RGB)/255
    plt.subplot(2, top_k+1, 1)
    plt.imshow(query_img)
    #plt.title("Query")
    plt.title(image_files[query_idx])
    plt.axis('off')
    
    # Top_k images
    for i, idx in enumerate(top_indices):
        img = cv2.cvtColor(cv2.imread(os.path.join("./dataset", image_files[idx])), cv2.COLOR_BGR2RGB)/255
        plt.subplot(2, top_k+1, i+2)
        plt.imshow(img)
        #plt.title(f"Rank {i+1}")
        plt.title(image_files[idx])
        plt.axis('off')
    #plt.show()
    
    #plt.figure(figsize=(15,4))
    plt.subplot(2, top_k+1, top_k+2)
    plt.bar(range(histograms[query_idx].size), histograms[query_idx], width=1.5)
    #plt.title("Query histogram")
    plt.title(f"{distance}: {dists[query_idx]:.4f}")
    plt.ylim(0, 1)
    
    for i, idx in enumerate(top_indices):
        plt.subplot(2, top_k+1, top_k+i+3)
        plt.bar(range(histograms[idx].size), histograms[idx], width=1.5)
        #plt.title(f"Rank {i+1} hist")
        plt.title(f"{distance}: {dists[idx]:.4f}")
        plt.ylim(0, 1)
    plt.show()


image_files, grad_features = compute_save_gradient_features("./dataset", sigma=1, grid_size=8, n_angle_bins=8)
query_idx = 19

for method in ["l2", "chi2", "intersection", "hellinger"]:
   retrieve_similar_images(query_idx, image_files, grad_features, distance=method, top_k=5)

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 2
#a)
def findedges(image, sigma, theta):
    image_m, _ = gradient_magnitude(image, sigma)

    image_out = np.where(image_m >= theta, image_m, 0)
    return image_out

sigma = 2
image = cv2.imread("./images/museum.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
theta = 0.10
image1 = findedges(image, sigma, theta)
theta = 0.5
image5 = findedges(image, sigma, theta)
theta = 0.9
image9 = findedges(image, sigma, theta)

plt.figure(figsize=(15, 8))
plt.suptitle("sigma = 1")
plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(image1, cmap="gray")
plt.title("theta = 0.1")
plt.axis('off')
plt.subplot(2, 2, 3)
plt.imshow(image5, cmap="gray")
plt.title("theta = 0.5")
plt.axis('off')
plt.subplot(2, 2, 4)
plt.imshow(image9, cmap="gray")
plt.title("theta = 0.9")
plt.axis('off')

plt.show()

#b)
def non_max_suppression(magnitude, angle):
    H, W = magnitude.shape

    # pad magnitude in angle 
    mag = np.pad(magnitude, pad_width=1, mode="reflect")
    ang = np.pad(angle, pad_width=1, mode="reflect")

    #radian -> stopinje (0, 360)
    ang = np.rad2deg(ang % (2*np.pi))

    nms = np.zeros_like(magnitude)

    for i in range(1, H+1):
        for j in range(1, W+1):

            q = 0
            r = 0

            a = ang[i, j]

            # 0 direction
            if (0 <= a < 22.5) or (337.5 <= a < 360):
                q = mag[i, j+1]
                r = mag[i, j-1]

            # 45 direction
            elif (22.5 <= a < 67.5):
                q = mag[i-1, j+1]
                r = mag[i+1, j-1]

            # 90 direction
            elif (67.5 <= a < 112.5):
                q = mag[i-1, j]
                r = mag[i+1, j]

            # 135 direction
            elif (112.5 <= a < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            # 180 direction (same as 0)
            elif (157.5 <= a < 202.5):
                q = mag[i, j-1]
                r = mag[i, j+1]

            # 225 direction (same as 45)
            elif (202.5 <= a < 247.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]

            # 270 direction (same as 90)
            elif (247.5 <= a < 292.5):
                q = mag[i+1, j]
                r = mag[i-1, j]

            # 315 direction (same as 135)
            elif (292.5 <= a < 337.5):
                q = mag[i+1, j+1]
                r = mag[i-1, j-1]

            # rezultat brez paddanja -> zato -1
            if mag[i, j] >= q and mag[i, j] >= r:
                nms[i-1, j-1] = mag[i, j]

    return nms



#c)
def hysteresis(nms, t_low, t_high):
    strong = nms >= t_high
    weak = (nms >= t_low) & (nms < t_high)

    # naredim labelmap
    # 255 = strong edge, 128 = weak edge, 0 = non-edge
    edges = np.zeros_like(nms, dtype=np.uint8)
    edges[strong] = 255
    edges[weak] = 128

    # najdi povezane komponente med weak edges + strong
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)

    # oznaci tiste, ki vsebujejo vsaj en strong edge
    final_edges = np.zeros_like(edges, dtype=np.uint8)
    for i in range(1, num_labels):
        component_mask = labels == i
        if np.any(strong[component_mask]):
            final_edges[component_mask] = 255

    return final_edges


sigma = 1.5
thr = 0.16
low = 0.04
image = cv2.imread("./images/museum.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
m, fi = gradient_magnitude(image, sigma)
image_edge_m = findedges(image, sigma, thr)
nms = non_max_suppression(image_edge_m, fi)
out = hysteresis(nms, low, thr)

image_c = cv2.imread("./images/museum.jpg", cv2.IMREAD_GRAYSCALE)
built_in_canny = cv2.Canny(image_c, 254, 255)

plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(image_edge_m, cmap="gray")
plt.title("Thresholded (thr = 0.16)")
plt.axis('off')
plt.subplot(2, 3, 4)
plt.imshow(nms, cmap="gray")
plt.title("Nonmax. supp. (thr = 0.16)")
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(out, cmap="gray")
plt.title("Hysteresis (high = 0.16, low = 0.04)")
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(built_in_canny, cmap="gray")
plt.title("Built in Canny")
plt.axis('off')

plt.show()

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 3
#a)
def canny_david(image, low, high, sigma=2):
    """
    high je uzet kr kot theta v findedges
    """
    sigma = 2
    _, fi = gradient_magnitude(image, sigma)
    image_edge_m = findedges(image, sigma, high)
    nms = non_max_suppression(image_edge_m, fi)
    out = hysteresis(nms, low, high)
    return out

n = 100
acc_size = 300
p = (50, 90)
diag_len = np.hypot(n, n)
rho_max = int(diag_len)
rhos = np.linspace(-rho_max, rho_max, acc_size)
thetas = np.deg2rad(np.linspace(-90, 90, acc_size))

H = np.zeros((acc_size, acc_size))

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(np.zeros((n, n)), cmap="gray")
for theta_idx, theta in enumerate(thetas):
    # zracunaj rho za current theta
    rho = p[0] * np.cos(theta) + p[1] * np.sin(theta)
    rho_idx = int(np.round((rho + rho_max) * (acc_size - 1) / (2 * rho_max)))
            
    if 0 <= rho_idx < acc_size and 0 <= theta_idx < acc_size:
        H[rho_idx, theta_idx] += 1
    a3.draw_line(rho, theta, n, n, clr="g")

plt.subplot(1, 2, 2)
plt.imshow(H, cmap="gray")
plt.show()

#b)
def hough_find_lines(image, num_theta_bins, num_rho_bins):
    acc = np.zeros((num_rho_bins, num_theta_bins))
    H, W = image.shape
    diag_len = np.hypot(W, H)
    rho_max = int(diag_len)
    
    thetas = np.deg2rad(np.linspace(-90, 90, num_theta_bins))
    rhos = np.linspace(-rho_max, rho_max, num_rho_bins)

    ys, xs = np.nonzero(image)

    for x, y in zip(xs, ys):
        for theta_idx, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            # pretvori rho v index
            rho_idx = int(np.round((rho + rho_max) * (num_rho_bins - 1) / (2 * rho_max)))
            
            if 0 <= rho_idx < num_rho_bins:
                acc[rho_idx, theta_idx] += 1

    return acc

def hough_find_lines_fast(image, num_theta_bins, num_rho_bins):
    H_img, W_img = image.shape
    diag_len = np.hypot(W_img, H_img)
    rho_max = int(diag_len)

    # theta and rho vectors
    thetas = np.deg2rad(np.linspace(-90, 90, num_theta_bins))
    rhos = np.linspace(-rho_max, rho_max, num_rho_bins)
    acc = np.zeros((num_rho_bins, num_theta_bins), dtype=np.uint64)

    ys, xs = np.nonzero(image)  # edge pixel koordinate

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for x, y in zip(xs, ys):
        rho_vals = x * cos_t + y * sin_t
        #rho je lahko od -rho_max do +rho_max, naprej spravimo vse na pozitivne, potem pa pomnožimo in delimo z 2*rho_max -> lahko bi obrnil vrstni red in bi najprej dobil decimalko in potem množil z bini
        rho_idxs = np.round((rho_vals + rho_max) * (num_rho_bins - 1) / (2 * rho_max)).astype(int)
        valid = (rho_idxs >= 0) & (rho_idxs < num_rho_bins)
        acc[rho_idxs[valid], np.arange(num_theta_bins)[valid]] += 1

    return acc


n = 200
test_image = np.zeros((100, 100))
test_image[10, 10] = 1
test_image[10, 20] = 1
acc1 = hough_find_lines_fast(test_image, n, n)

oneline = cv2.imread("./images/oneline.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
oneline_edge = canny_david(oneline, 0.04, 0.16)
acc2 = hough_find_lines_fast(oneline_edge, n, n)

rectangle = cv2.imread("./images/rectangle.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
rectangle_edge = canny_david(rectangle, 0.04, 0.16)
acc3 = hough_find_lines_fast(rectangle_edge, n, n)


plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(acc1)
plt.title("Synthetic")
plt.subplot(1, 3, 2)
plt.imshow(acc2)
plt.title("oneline.png")
plt.subplot(1, 3, 3)
plt.imshow(acc3)
plt.title("rectangle.png")

plt.show()

#c)
def nonmaxima_suppression_box(acc, k=5):
    N = k // 2
    height, width = acc.shape
    suppressed = np.zeros_like(acc, dtype=np.float64)

    # paddanje
    padded = np.pad(acc, ((N, N), (N, N)), mode='constant', constant_values=0)

    ys, xs = np.nonzero(acc)
    # gremo cez akumulator
    for i, j in zip(ys, xs):
        window = padded[i:i+k, j:j+k]
        max_val = window.max()
        
        # center ostane ce je enak max
        if padded[i+N, j+N] == max_val and max_val > 0:
            # vse ostale na 0
            window[:, :] = 0
            # ohrani center
            window[N, N] = max_val
            suppressed[i, j] = max_val
            # posodobi padded, da vidimo suppres v naslednji generaciji
            padded[i:i+k, j:j+k] = window

    return suppressed


#d)
def extract_lines(image, num_theta_bins, num_rho_bins, threshold, acc):
    k = 5
    acc_nms = nonmaxima_suppression_box(acc, k)

    H_img, W_img = image.shape
    diag_len = np.hypot(W_img, H_img)
    rho_max = int(diag_len)

    thetas = np.deg2rad(np.linspace(-90, 90, num_theta_bins))
    rhos = np.linspace(-rho_max, rho_max, num_rho_bins)

    ys, xs = np.nonzero(acc_nms >= threshold)
    
    lines = []
    for y, x in zip(ys, xs):
        rho = rhos[y]
        theta = thetas[x]
        lines.append((rho, theta))
    
    #z vektorji:
    #lines = [(rhos[y], thetas[x]) for y, x in zip(ys, xs)]

    return lines

n = 200
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(test_image, cmap="gray")
plt.title("Synthetic")
h, w = test_image.shape
lines1 = extract_lines(test_image, n, n, 2, acc1)
for rho, theta in lines1:
    a3.draw_line(rho, theta, h, w)

plt.subplot(1, 3, 2)
plt.imshow(oneline, cmap="gray")
plt.title("oneline.png")
h, w = oneline.shape
lines2 = extract_lines(oneline, n, n, 170, acc2)
for rho, theta in lines2:
    a3.draw_line(rho, theta, h, w)

plt.subplot(1, 3, 3)
plt.imshow(rectangle, cmap="gray")
plt.title("rectangle.png")
h, w = rectangle.shape
lines3 = extract_lines(rectangle, n, n, 170, acc3)
for rho, theta in lines3:
    a3.draw_line(rho, theta, h, w)

plt.show()

#e)
def extract_top_n_lines(image, num_theta_bins, num_rho_bins, n_top, acc):
    k = 5
    acc_nms = nonmaxima_suppression_box(acc, k)
    
    H_img, W_img = image.shape
    diag_len = np.hypot(W_img, H_img)
    rho_max = int(diag_len)
    
    thetas = np.deg2rad(np.linspace(-90, 90, num_theta_bins))
    rhos = np.linspace(-rho_max, rho_max, num_rho_bins)
    
    # Flatten accumulator (kopija + ena dimenzija) za sortiranje po glasovih
    votes = acc_nms.flatten()
    indices = np.argsort(votes)[::-1]
    top_indices = indices[:n_top]
    
    lines = []
    for idx in top_indices:
        rho_idx = idx // num_theta_bins
        theta_idx = idx % num_theta_bins


        if votes[idx] > 0:
            lines.append((rhos[rho_idx], thetas[theta_idx]))
    
    return lines


n = 400
sigma = 2
bricks_color = cv2.imread("./images/bricks.jpg", cv2.IMREAD_COLOR)  # BGR
bricks_color = cv2.cvtColor(bricks_color, cv2.COLOR_BGR2RGB)
bricks = cv2.cvtColor(bricks_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
bricks_edge = canny_david(bricks, 0.14, 0.36, sigma=1)
acc1 = hough_find_lines_fast(bricks_edge, n, n)
lines1 = extract_top_n_lines(bricks, n, n, n_top=10, acc=acc1)


building_color = cv2.imread("./images/building.jpg", cv2.IMREAD_COLOR)
building_color = cv2.cvtColor(building_color, cv2.COLOR_BGR2RGB)
building = cv2.cvtColor(building_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
building_edge = canny_david(building, 0.14, 0.26, sigma=2)
acc2 = hough_find_lines_fast(building_edge, n, n)
lines2 = extract_top_n_lines(building, n, n, n_top=10, acc=acc2)

pier_color = cv2.imread("./images/pier.jpg", cv2.IMREAD_COLOR)
pier_color= cv2.cvtColor(pier_color, cv2.COLOR_BGR2RGB)
pier = cv2.cvtColor(pier_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
pier_edge = canny_david(pier, 0.14, 0.36, sigma=3)
acc3= hough_find_lines_fast(pier_edge, n, n)
lines3 = extract_top_n_lines(pier, n, n, n_top=10, acc=acc3)

plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(acc1)
plt.title("bricks.jpg")
plt.subplot(2, 3, 4)
plt.imshow(bricks_color)
h, w = bricks.shape
for rho, theta in lines1:
    a3.draw_line(rho, theta, h, w)

plt.subplot(2, 3, 2)
plt.imshow(acc2)
plt.title("building.jpg")
plt.subplot(2, 3, 5)
plt.imshow(building_color)
h, w = building.shape
for rho, theta in lines2:
    a3.draw_line(rho, theta, h, w)

plt.subplot(2, 3, 3)
plt.imshow(acc3)
plt.title("pier.jpg")
plt.subplot(2, 3, 6)
plt.imshow(pier_color)
h, w = pier.shape
for rho, theta in lines3:
    a3.draw_line(rho, theta, h, w)

plt.show()


#f)
def hough_find_lines_angles(image, num_theta_bins, num_rho_bins, edge_angles, magnitude, angle_tolerance_deg=5):
    H_img, W_img = image.shape
    diag_len = np.hypot(W_img, H_img)
    rho_max = int(diag_len)

    # Parametri
    thetas = np.deg2rad(np.linspace(-90, 90, num_theta_bins))
    rhos = np.linspace(-rho_max, rho_max, num_rho_bins)
    acc = np.zeros((num_rho_bins, num_theta_bins), dtype=np.float64)

    ys, xs = np.nonzero(image)
    angle_tolerance = np.deg2rad(angle_tolerance_deg)

    for x, y in zip(xs, ys):
        phi = edge_angles[y, x]

        #spraviti moras v -pi/2 do pi/2, najprej pristejes pi/2 in das modulo pi -> dobis (0, pi), potem pa odstejes pi/2, da dobis pravokotno
        theta_expected = ((phi + np.pi/2) % np.pi) - np.pi/2

        theta_min = theta_expected - angle_tolerance
        theta_max = theta_expected + angle_tolerance

        # ce je min ali max cez, gres okoli, da jih zajames
        if theta_min < -np.pi/2:
            valid_indices = np.where((thetas >= theta_min + np.pi) | (thetas <= theta_max))[0]
        elif theta_max > np.pi/2:
            valid_indices = np.where((thetas >= theta_min) | (thetas <= theta_max - np.pi))[0]
        else:
            valid_indices = np.where((thetas >= theta_min) & (thetas <= theta_max))[0]

        # Voting
        for theta_idx in valid_indices:
            theta = thetas[theta_idx]
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int(np.round((rho + rho_max) * (num_rho_bins - 1) / (2 * rho_max)))

            if 0 <= rho_idx < num_rho_bins:
                acc[rho_idx, theta_idx] += magnitude[y, x]

    return acc

n = 400
sigma = 1
theta = 0.16
rectangle = cv2.imread("./images/rectangle.png", cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
rectangle_edges = canny_david(rectangle, 0.04, 0.16, sigma)
m, fi = gradient_magnitude(rectangle_edge, sigma)
acc_n = hough_find_lines_fast(rectangle_edge, n, n)

acc_o = hough_find_lines_angles(rectangle_edge, n, n, fi, m)
h, w = rectangle.shape


plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)
plt.imshow(acc_n)
plt.title("Normal")
plt.subplot(2, 2, 3)
plt.imshow(rectangle, cmap="gray")
lines = extract_lines(rectangle, n, n, 150, acc_n)
for rho, theta in lines:
    a3.draw_line(rho, theta, h, w)

plt.subplot(2, 2, 2)
plt.imshow(acc_o)
plt.title("Orientation")
plt.subplot(2, 2, 4)
plt.imshow(rectangle, cmap="gray")
lines = extract_lines(rectangle, n, n, 40, acc_o)
for rho, theta in lines:
    a3.draw_line(rho, theta, h, w)

plt.show()


#g)
def hough_circles_fixed_radius(image, radius):
    H, W = image.shape
    acc = np.zeros((H, W), dtype=np.float64)

    ys, xs = np.nonzero(image)

    for x, y in zip(xs, ys):
        for t in np.linspace(0, 2*np.pi, 360):
            a = int(round(x - radius * np.cos(t)))
            b = int(round(y - radius * np.sin(t)))
            if 0 <= a < W and 0 <= b < H:
                acc[b, a] += 1

    return acc

radius = 45
coins_color = cv2.imread("./images/coins.jpg", cv2.IMREAD_COLOR)  # BGR
coins_color = cv2.cvtColor(coins_color, cv2.COLOR_BGR2RGB)
coins = cv2.cvtColor(coins_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255

edges_coins = canny_david(coins, 0.1, 0.2)
acc1 = hough_circles_fixed_radius(edges_coins, radius)

centers_coins = np.argwhere(acc1 > np.percentile(acc1, 99.99))

radius = 50
eclipse_color = cv2.imread("./images/eclipse.jpg", cv2.IMREAD_COLOR)
eclipse_color = cv2.cvtColor(eclipse_color, cv2.COLOR_BGR2RGB)
eclipse = cv2.cvtColor(eclipse_color, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255

edges_eclipse = canny_david(eclipse, 0.04, 0.2)
acc2 = hough_circles_fixed_radius(edges_eclipse, radius)

centers_eclipse = np.argwhere(acc2 > np.percentile(acc2, 99.9))

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(acc1)
plt.subplot(2, 2, 3)
plt.imshow(coins_color)
for b, a in centers_coins:
    circle = plt.Circle((a, b), radius, color='g', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(acc2)
plt.subplot(2, 2, 4)
plt.imshow(eclipse_color)
for b, a in centers_eclipse:
    circle = plt.Circle((a, b), radius, color='g', fill=False, linewidth=1)
    plt.gca().add_patch(circle)
plt.axis("off")

plt.show()


#h)
def extract_top_n_lines_normalized(image, num_theta_bins, num_rho_bins, n_top, acc):
    H_img, W_img = image.shape
    diag_len = np.hypot(W_img, H_img)
    rho_max = int(diag_len)
    
    thetas = np.deg2rad(np.linspace(-90, 90, num_theta_bins))
    rhos = np.linspace(-rho_max, rho_max, num_rho_bins)
    
    acc_norm = np.zeros_like(acc, dtype=np.float64)
    
    for t_idx, theta in enumerate(thetas):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for r_idx, rho in enumerate(rhos):
            points = []

            # y=0
            if cos_t != 0:
                x = (rho - 0*sin_t)/cos_t
                if 0 <= x < W_img: points.append((x, 0))
            # y=H-1
            if cos_t != 0:
                x = (rho - (H_img-1)*sin_t)/cos_t
                if 0 <= x < W_img: points.append((x, H_img-1))
            # x=0
            if sin_t != 0:
                y = (rho - 0*cos_t)/sin_t
                if 0 <= y < H_img: points.append((0, y))
            # x=W-1
            if sin_t != 0:
                y = (rho - (W_img-1)*cos_t)/sin_t
                if 0 <= y < H_img: points.append((W_img-1, y))

            if len(points) >= 2:
                # vzemi najbolj odaljeni točki
                max_dist = 0
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        d = np.hypot(points[j][0]-points[i][0], points[j][1]-points[i][1])
                        if d > max_dist:
                            max_dist = d
                
                acc_norm[r_idx, t_idx] = acc[r_idx, t_idx] / max_dist
    
    votes = acc_norm.flatten()
    indices = np.argsort(votes)[::-1]
    top_indices = indices[:n_top]
    
    lines = []
    for idx in top_indices:
        rho_idx = idx // num_theta_bins
        theta_idx = idx % num_theta_bins
        if votes[idx] > 0:
            lines.append((rhos[rho_idx], thetas[theta_idx]))
    return lines


img = cv2.imread("./images/rectangle.png", cv2.IMREAD_GRAYSCALE)/255
edges = canny_david(img, 0.04, 0.16, sigma=1)
acc = hough_find_lines_fast(edges, 400, 400)

lines_unnormalized = extract_top_n_lines(img, 400, 400, 10, acc)
lines_normalized = extract_top_n_lines_normalized(img, 400, 400, 10, acc)


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
for rho, theta in lines_unnormalized:
    a3.draw_line(rho, theta, *img.shape)
plt.title("Unnormalized")

plt.subplot(1,2,2)
plt.imshow(img, cmap='gray')
for rho, theta in lines_normalized:
    a3.draw_line(rho, theta, *img.shape)
plt.title("Normalized")
plt.show()
