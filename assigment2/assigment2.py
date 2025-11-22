import a2_utils as a2
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 1 
#a)
#Glej bralnik

#b)
def simple_convolution(signal_I, kernel):
    N = (len(kernel) - 1) // 2
    signal_Out = np.zeros(len(signal_I) - 2 * N, dtype=np.float64)

    for i in range (N, len(signal_I) - N - 1):
        signal_Out[i - N] = (signal_I[i - N : i + N + 1] * kernel[::-1]).sum()

    return signal_Out

signal_I = a2.read_data("signal.txt")
kernel = a2.read_data("kernel.txt")

convolve = simple_convolution(signal_I, kernel)

#število vrstic -> -1 pomeni da pogrunta sam, število stolpcev....lahko uporabis np.expand_dims(kernel, axis=1) namesto tega
kernel_2d = kernel.reshape(-1, 1)
#ddepth=-1 ohrani enako globino slike
convolveLib = cv2.filter2D(signal_I, ddepth=-1, kernel=kernel_2d)

plt.figure(figsize=(15, 8))
plt.title("Convolution")
plt.plot(signal_I, color="b", label="Original Signal")
plt.plot(kernel, color="orange", label="Original Kernel")
plt.plot(convolve, color="g", label="My Simple Convolution")
plt.plot(convolveLib, color="r", label="CV2 Library Convolution")

plt.legend()

plt.show()

#QUESTION: Can you recognize the shape of the kernel? What is the sum of the elements in the kernel? How does the kernel affect the signal?
#          Jedro ima obliko Gaussove porazdelitve (zvonasta oblika).
#          Vsota vseh elementov je 1, zato signal zgladi (zmanjša šum) brez spreminjanja svetlosti/amplitude.


#c
def complex_convolution(signal_I, kernel):
    N = (len(kernel) - 1) // 2

    signal_start = signal_I[N-1::-1]
    signal_end = signal_I[:N-1:-1]
    signal_final = np.concatenate((signal_start, signal_I, signal_end))
    signal_Out = np.zeros(len(signal_I), dtype=np.float64)

    for i in range (len(signal_I)):
        j = i + N
        signal_Out[i] = (signal_final[j - N : j + N + 1] * kernel[::-1]).sum()

    return signal_Out

signal_I = a2.read_data("signal.txt")
kernel = a2.read_data("kernel.txt")

convolve = complex_convolution(signal_I, kernel)

kernel_2d = kernel.reshape(-1, 1)
convolveLib = cv2.filter2D(signal_I, ddepth=-1, kernel=kernel_2d)

plt.figure(figsize=(15, 8))
plt.title("Convolution")
plt.plot(signal_I, color="b", label="Original Signal")
plt.plot(kernel, color="orange", label="Original Kernel")
plt.plot(convolve, color="g", label="My Complex Convolution")
plt.plot(convolveLib, color="r", label="CV2 Library Convolution")

plt.legend()

plt.show()


#d
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

kernel1 = gauss(0.5)
kernel2 = gauss(1)
kernel3 = gauss(2)
kernel4 = gauss(3)
kernel5 = gauss(4)

plt.figure(figsize=(15, 8))
plt.title("Gaussian Kernel")
plt.plot(np.arange(-2, 3), kernel1, color="b", label="sigma = 0.5")
plt.plot(np.arange(-4, 5), kernel2, color="orange", label="sigma = 1")
plt.plot(np.arange(-7, 8), kernel3, color="g", label="sigma = 2")
plt.plot(np.arange(-10, 11), kernel4, color="r", label="sigma = 3")
plt.plot(np.arange(-13, 14), kernel5, color="purple", label="sigma = 4")

plt.legend()

plt.show()

#QUESTION: The figure below shows two kernels (a) and (b) as well as signal (c). Sketch (do not focus on exact proportions of your drawing, but rather on the understanding
#          of what you are doing) the resulting convolved signal of the given input signal and each kernel.
#          Glej bralnik

#e
signal_I = a2.read_data("signal.txt")
kernel1 = gauss(2)
kernel2 = [0.1, 0.6, 0.4]

temp = complex_convolution(signal_I, kernel1)
conv1 = complex_convolution(temp, kernel2)

temp = complex_convolution(signal_I, kernel2)
conv2 = complex_convolution(temp, kernel1)

temp = complex_convolution(kernel1, kernel2)
conv3 = complex_convolution(signal_I, temp)

plt.figure(figsize=(15, 6))
plt.suptitle("Associativity Of Convolution")
plt.subplot(1, 4, 1)
plt.plot(signal_I)
plt.title("s")
plt.subplot(1, 4, 2)
plt.plot(conv1)
plt.title("(s * k1) * k2")
plt.subplot(1, 4, 3)
plt.plot(conv2)
plt.title("(s * k2) * k1")
plt.subplot(1, 4, 4)
plt.plot(conv3)
plt.title("s * (k1 * k2)")

plt.tight_layout()
plt.show()


#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 2

#a)
def gaussfilter(image, sigma):
    filter = np.expand_dims(gauss(sigma), axis=1)

    image2 = cv2.filter2D(image, ddepth=-1, kernel=filter)
    image3 = cv2.filter2D(image2, ddepth=-1, kernel=filter.T)
    return image3

#prebere sliko v formatu uint8
lena = cv2.imread("./images/lena.png", cv2.IMREAD_GRAYSCALE).astype(np.float64)
sigma = 2

lena2 = a2.gauss_noise(lena / 255)
lena3 = a2.sp_noise(lena / 255)

lena2g = gaussfilter(lena2, sigma)
lena3g = gaussfilter(lena3, sigma)

plt.figure(figsize=(15, 6))
plt.subplot(2, 3, 1)
plt.imshow(lena, cmap="gray")
plt.title("Original")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(lena2, cmap="gray")
plt.title("Gaussian noise")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(lena3, cmap="gray")
plt.title("Salt-pepper noise")
plt.axis('off')
plt.subplot(2, 3, 5)
plt.imshow(lena2g, cmap="gray")
plt.title("Filtered Gaussian noise")
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(lena3g, cmap="gray")
plt.title("Filtered Salt-pepper noise")
plt.axis('off')

plt.show()

#QUESTION: Which noise is better removed using the Gaussian filter?
#           Gaussov filter bolje odstrani Gaussov šum, ker deluje kot nizkopasovni filter in zgladi naključne majhne spremembe intenzitete pikslov, pri čemer ohranja strukturo slike.
#           Pri sol in poper šumu pa Gaussov filter ne učinkuje dobro, saj redke ekstremne vrednosti le zamegli v sosednje piksle, ne pa popolnoma odstrani,
#           za to je primernejši mediana filter.
#

#b)
sigma = 2
fox = cv2.imread("./images/fox.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float64) /255
fox_blurred = gaussfilter(fox, sigma)

fox_sharp = 2*fox - fox_blurred

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float64)

fox_sharp2 = cv2.filter2D(fox, ddepth=-1, kernel=sharpen_kernel)

plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(fox, cmap="gray", vmin=0, vmax=1)
plt.title("Original")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(fox_sharp, cmap="gray", vmin=0, vmax=1)
plt.title("Sharpened Using Gaussian Blur")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(fox_sharp2, cmap="gray", vmin=0, vmax=1)
plt.title("Sharpened Using Sharpening Kernel")
plt.axis('off')

plt.tight_layout()
plt.show()

#c)
sigma = 2
width = 3
def simple_median(signal_I, w):
    N = w // 2

    signal_start = signal_I[N-1::-1]
    signal_end = signal_I[:N-1:-1]
    signal_final = np.concatenate((signal_start, signal_I, signal_end))
    signal_Out = np.zeros(len(signal_I), dtype=np.float64)

    for i in range (len(signal_I)):
        j = i + N
        signal_Out[i] = np.median(signal_final[j - N : j + N + 1])

    return signal_Out


signal_I = a2.read_data("signal2.txt")
signal_sp = a2.read_data("signal2corrupted.txt")
singal_sp = np.expand_dims(signal_sp, 1)

signalGauss = cv2.filter2D(signal_sp, ddepth=-1, kernel=gauss(sigma))
signalMedian2 = simple_median(signal_sp, width)
width = 5
signalMedian5 = simple_median(signal_sp, width)
width = 11
signalMedian11 = simple_median(signal_sp, width)

plt.figure(figsize=(18, 5))
plt.subplot(1, 6, 1)
plt.plot(signal_I)
plt.title("Original")
plt.ylim(0, 3)
plt.subplot(1, 6, 2)
plt.plot(signal_sp)
plt.title("Corrupted")
plt.ylim(0, 3)
plt.subplot(1, 6, 3)
plt.plot(signalGauss)
plt.title("Gauss")
plt.ylim(0, 3)
plt.subplot(1, 6, 4)
plt.plot(signalMedian2)
plt.title("Median w=2")
plt.ylim(0, 3)
plt.subplot(1, 6, 5)
plt.plot(signalMedian5)
plt.title("Median w=5")
plt.ylim(0, 3)
plt.subplot(1, 6, 6)
plt.plot(signalMedian11)
plt.title("Median w=11")
plt.ylim(0, 3)

plt.tight_layout()
plt.show()

#QUESTION: Which filter performs better at this specific task? In comparison to Gaussian filter that can be applied multiple times in any order, does the order
#          matter in case of median filter? What is the name of filters like this?

#          Median filter je boljši, ker zelo lepo pobriše sol-in-poper šum, ne da bi toliko zameglil sliko kot Gauss.
#          Pri median filtru vrstni red ni pomemben, takšnim filtrom pa rečemo idempotentni filtri.

#d)
def complex_median(image, w):
    N = w // 2

    height, width = image.shape

    padded = np.zeros((height, width),  dtype=np.float64)
    # Pad image with border replication
    padded = np.pad(image, ((N, N), (N, N)), mode='symmetric')
    
    # Output image
    filtered = np.zeros((height, width),  dtype=np.float64)
    
    # Loop over all pixels
    for i in range(height):
        for j in range(width):
            window = padded[i:i+w, j:j+w]
            filtered[i, j] = np.median(window)
    
    return filtered

width = 5

lena2m = complex_median(lena2, width)
lena3m = complex_median(lena3, width)

plt.figure(figsize=(15, 10))
plt.subplot(3, 3, 1)
plt.imshow(lena, cmap="gray")
plt.title("Original")
plt.axis('off')
plt.subplot(3, 3, 2)
plt.imshow(lena2, cmap="gray")
plt.title("Gaussian noise")
plt.axis('off')
plt.subplot(3, 3, 3)
plt.imshow(lena3, cmap="gray")
plt.title("Salt-pepper noise")
plt.axis('off')
plt.subplot(3, 3, 5)
plt.imshow(lena2g, cmap="gray")
plt.title("Using Gaussian Filter")
plt.axis('off')
plt.subplot(3, 3, 6)
plt.imshow(lena3g, cmap="gray")
plt.title("Using Gaussian Filter")
plt.axis('off')
plt.subplot(3, 3, 8)
plt.imshow(lena2m, cmap="gray")
plt.title("Using Median Filte")
plt.axis('off')
plt.subplot(3, 3, 9)
plt.imshow(lena3m, cmap="gray")
plt.title("Using Median Filter")
plt.axis('off')

plt.tight_layout()
plt.show()

#QUESTION: What is the computational complexity of the Gaussian filter operation? How about the median filter? What does it depend on? Describe the computational
#          complexity using the O(.) notation (you can assume n log n complexity for sorting).
#          Kompleksnost Gaussovega filtra je O(n*k), kjer je n število pikslov in k velikost jedra, ker za vsako točko seštejemo vrednosti okoli nje. 
#          Mediana filter je O(n*k*log k), ker moramo za vsako točko sortirati k sosednjih vrednosti, da dobimo mediano.
#

#______________________________________________________________________________________________________________________________________________________________
#EXERCISE 3

#a)
def myhist3(image, n_bins):
    H = np.zeros((n_bins,n_bins,n_bins))

    #st vrstic, st stolpcev
    I_1D = image.reshape(-1,3)

    for i in range (I_1D.shape[0]):
        #R
        bin_index0 = int(I_1D[i][0] * n_bins)
        if bin_index0 == n_bins:  # To se zgodi ob I_1D[i] == 1.0 -> patch za max vrednost
            bin_index0 = n_bins - 1
        #G
        bin_index1 = int(I_1D[i][1] * n_bins)
        if bin_index1 == n_bins:
            bin_index1 = n_bins - 1
        #B
        bin_index2 = int(I_1D[i][2] * n_bins)
        if bin_index2 == n_bins:
            bin_index2 = n_bins - 1
        H[bin_index0, bin_index1, bin_index2] += 1

    H /= I_1D.shape[0]
    return H

#debug
#lena = cv2.imread("./images/lena.png").astype(np.float64) / 255
#hist = myhist3(lena, 10)
#kr neka vsota
#plt.bar(range(10), hist.sum(axis=(1,2)))
#plt.show()

#poskusimo brez for loopa

# Flatten image: pixels is N x 3 with N = H*W pixels.
# Compute bins: Multiply by n_bins and floor to get integer bin indices for R, G, B.
# Clip values: Make sure max value 1.0 goes to the last bin.
# Flatten 3D indices to 1D: np.ravel_multi_index converts (R_bin, G_bin, B_bin) to a single index.
# Count occurrences: np.bincount counts how many pixels fall into each 3D bin.
# Reshape and normalize: Histogram has shape (n_bins, n_bins, n_bins) and sums to 1.

def myhist3_1(image, n_bins):
    H, W, C = image.shape
    assert C == 3, "Image must have 3 channels"

    pixels = image.reshape(-1, 3)

    bins = np.floor(pixels * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins-1)  # handle edge case where value=1.0

    #ravel vrne le pogled, če bi dal reshape(-1) bi naredil kopijo
    #ravel_multi_index zagotavlja unikatnost 1D indeksa za vsako kombinacijo 3D binov.
    #parametri: multi_indextuple of array_like -> A tuple of integer arrays, one array for each dimension.
                #dims: tuple of ints -> The shape of array into which the indices from multi_index apply.

    idx = np.ravel_multi_index((bins[:,0], bins[:,1], bins[:,2]), dims=(n_bins, n_bins, n_bins))

    #stejemo pojavitve
    H3 = np.bincount(idx, minlength=n_bins**3).reshape((n_bins, n_bins, n_bins))

    H3 = H3 / pixels.shape[0]

    return H3

#b)
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
        
#c)
n = 8
obj1 = cv2.cvtColor(cv2.imread("./dataset/object_01_1.png"), cv2.COLOR_BGR2RGB).astype(np.float64)/255
hist1 = myhist3(obj1, n)
obj2 = cv2.cvtColor(cv2.imread("./dataset/object_02_1.png"), cv2.COLOR_BGR2RGB).astype(np.float64)/255
hist2 = myhist3(obj2, n)
obj3 = cv2.cvtColor(cv2.imread("./dataset/object_03_1.png"), cv2.COLOR_BGR2RGB).astype(np.float64)/255
hist3 = myhist3(obj3, n)

plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.imshow(obj1)
plt.title("object_01_1")
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(obj2)
plt.title("object_02_1e")
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(obj3)
plt.title("object_03_1")
plt.axis('off')
plt.subplot(2, 3, 4)
plt.bar(range(512), hist1.ravel(), width=1.5)
plt.ylim(0, 1)
plt.subplot(2, 3, 5)
plt.bar(range(512), hist2.ravel(), width=1.5)
plt.ylim(0, 1)
plt.subplot(2, 3, 6)
plt.bar(range(512), hist3.ravel(), width=1.5)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

dist_1_2 = compare_histograms(hist1, hist2, "l2")
dist_1_3 = compare_histograms(hist1, hist3, "l2")
print("Euclidean distance between histograms of object 1 and 2 is:", dist_1_2)
print("Euclidean distance between histograms of object 1 and 3 is:", dist_1_3)
dist_1_2 = compare_histograms(hist1, hist2, "chi2")
dist_1_3 = compare_histograms(hist1, hist3, "chi2")
print("Chi-square distance between histograms of object 1 and 2 is:", dist_1_2)
print("Chi-square distance between histograms of object 1 and 3 is:", dist_1_3)
dist_1_2 = compare_histograms(hist1, hist2, "intersection")
dist_1_3 = compare_histograms(hist1, hist3, "intersection")
print("Intersection between histograms of object 1 and 2 is:", dist_1_2)
print("Intersection between histograms of object 1 and 3 is:", dist_1_3)
dist_1_2 = compare_histograms(hist1, hist2, "hellinger")
dist_1_3 = compare_histograms(hist1, hist3, "hellinger")
print("Hellinger distance between histograms of object 1 and 2 is:", dist_1_2)
print("Hellinger distance between histograms of object 1 and 3 is:", dist_1_3)


# QUESTION: Which image (object_02_1.png or object_03_1.png) is more similar to image object_01_1.png considering the L2 distance? How about the other three distances? 
#           We can see that all three histograms contain a strongly expressed component (one bin has a much higher value than the others). Which color does this bin represent?
#           Pri razdalji L2 je sliki object_01_1.png bolj podobna object_03_1.png, saj je razdalja med njima manjša (v bistvu pri vseh razdaljah)
#           V vseh histogramih vidimo, da je zelo močno reprezentirana rumena in pa črna barva. Še najbolj črna, saj je to ozadje slike.
#


#d)
def compute_save_histograms(image_dir, n_bins=8, save_path="histograms.npy"):

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    histograms = []

    for f in image_files:
        img_path = os.path.join(image_dir, f)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float64)/255
        hist = myhist3(img, n_bins).ravel()  # flatten to 1D
        histograms.append(hist)

    histograms = np.array(histograms)
    np.save(save_path, histograms)
    print(f"Saved {len(histograms)} histograms to {save_path}")
    return image_files, histograms


def retrieve_similar_images(query_idx, image_files, histograms, distance="l2", top_k=5):

    query_hist = histograms[query_idx]
    
    # razdalja do vseh
    dists = [compare_histograms(query_hist, hist, distance) for hist in histograms]

    # Sortiraj indices by increasing distance (smaller = more similar) -> Returns the indices that would sort an array
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
    
    # Plot their histograms
    #plt.figure(figsize=(15,4))
    plt.subplot(2, top_k+1, top_k+2)
    plt.bar(range(histograms[query_idx].size), histograms[query_idx], width=1.5)
    #plt.title("Query histogram")
    plt.title(f"{distance}: {dists[query_idx]:.4f}")
    
    for i, idx in enumerate(top_indices):
        plt.subplot(2, top_k+1, top_k+i+3)
        plt.bar(range(histograms[idx].size), histograms[idx], width=1.5)
        #plt.title(f"Rank {i+1} hist")
        plt.title(f"{distance}: {dists[idx]:.4f}")
    plt.show()


# histogrami za vse slike v ./dataset
image_files, histograms = compute_save_histograms("./dataset", n_bins=8, save_path="histograms.npy")

# iskana slika
query_idx = 19

for method in ["l2", "chi2", "intersection", "hellinger"]:
    #print(f"Using distance: {method}")
    retrieve_similar_images(query_idx, image_files, histograms, distance=method, top_k=5)

#Question: Which distance is in your opinion best suited for image retrieval? How does the retrieved sequence change if you use a different number of bins? Is the
#          execution time affected by the number of bins?
#          Za iskanje slik je običajno najbolj primerna Chi2 ali Hellinger razdalja, ker bolje upoštevata obliko histogramov in razliko v barvnih porazdelitvah.
#          Če spremenimo število binov, se lahko vrstni red najdenih podobnih slik spremeni, ker več binov pomeni bolj natančen, a tudi bolj "hrupen" histogram.
#          Več binov poveča čas izvajanja, saj je treba primerjati več vrednosti histogramov.
#

#e)
def plot_histogram_distances(query_idx, histograms, top_k=5, distance="l2"):

    dists = np.array([compare_histograms(histograms[query_idx], hist, distance) for hist in histograms])
    indices = np.arange(len(histograms))
    
    # nesortirano
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    #črta + krogeci
    plt.plot(indices, dists, '-o', label="Distance")
    
    # Highlight top-k najbolj podobnih
    top_indices = np.argsort(dists)[:top_k+1]  # include query itself
    #rdeče + krogeci
    plt.plot(top_indices, dists[top_indices], 'ro', markersize=8, label="Top-k")
    
    plt.title(f"Unsorted distances ({distance})")
    plt.xlabel("Image index")
    plt.ylabel("Distance to query")
    plt.legend()
    
    # sortirano
    plt.subplot(1, 2, 2)
    sorted_indices = np.argsort(dists)
    plt.plot(dists[sorted_indices], '-o', label="Distance")
    
    # Highlight top-k
    plt.plot(sorted(dists[sorted_indices[:top_k+1]]), 'ro', markersize=8, label="Top-k")
    
    plt.title(f"Sorted distances ({distance})")
    plt.xlabel("Sorted image rank")
    plt.ylabel("Distance to query")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


for method in ["l2", "chi2", "intersection", "hellinger"]:
    #print(f"Using distance: {method}")
    plot_histogram_distances(query_idx, histograms, top_k=5, distance=method)


#f)
# vsota vseh histogramov bin-wise
freq_hist = np.sum(histograms, axis=0)

plt.figure(figsize=(15, 4))
plt.bar(range(len(freq_hist)), freq_hist)
plt.title("Frequency of bins across the dataset")
plt.xlabel("Bin index")
plt.ylabel("Total count")
plt.show()

# kateri bini prevladujejo
dominant_bins = np.argsort(freq_hist)[-10:]  # top 10 bins
print("Top 10 dominant bins:", dominant_bins)

alpha = 5.0
weights = np.exp(-alpha * freq_hist)

#za boljšo natančnot normaliziramo -> nimamo ogromnih številk in potem malih uteži pri eksponentu
freq_hist_norm = freq_hist / np.sum(freq_hist)
weights = np.exp(-alpha * freq_hist_norm)

weighted_histograms = histograms * weights  # broadcast multiplication
#sedaj še vsak histogram posebej normaliziramo -> keepdims uporabi zato, da lahko deli vstico po vrstico pravilno, torej vsak histogram s svojo vsoto
#ohrani results shape (N,1), da se lahko broadcast deli (N, 512) z (N, 1)
weighted_histograms /= np.sum(weighted_histograms, axis=1, keepdims=True)

for method in ["l2", "chi2", "intersection", "hellinger"]:
    #print(f"Using distance: {method}")
    retrieve_similar_images(query_idx, image_files, histograms=weighted_histograms, distance=method, top_k=5)

# Weighting effect
#   - Da, weighting pomaga zmanjšati dominanco ozadja (nebo, trava itd.)
#   - Iskanje se izboljša za predmete/objekte s podobnimi oblikami, a različnim ozadjem
#
# Did the weighting help with retrieving relevant results?
#   Recimo da je, ozadje sedaj manj vpliva na iskane slike (Ne prav veliko, saj je ozadje okoli vseh slik črno - torej enako)
