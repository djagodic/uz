import cv2
import numpy as np
from matplotlib import pyplot as plt
from UZ_utils import *

#----------------------------------
#EXERCISE 1
#a)

I = imread("./images/umbrellas.jpg")
imshow(I)
height, width, channels = I.shape
print("Višina:", height, "    Širina:", width, "    Channels:", channels)
print("Tip:", I.dtype)


#b)
I_b=np.copy(I)
I_float = I_b.astype(np.float64)
I_gray = np.zeros((height, width), dtype=float)
I_gray = (I_float[:,:,0] + I_float[:,:,1] + I_float[:,:,2]) / 3
#Alternative option: I_gray = np.mean(I, axis=2)
imshow(I_gray)


#c)
I_c=np.copy(I)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
#Ker gledam axis2 = 1 -> green channel
cutout1=I_c[130:260, 240:450, 1]
plt.imshow(cutout1, cmap="gray")
#Določim naslov in izklopim osi oz. številčenje ob strani
plt.title("Cutout 1")
plt.axis('off')

plt.subplot(1, 2, 2)
#axis2 = 1 -> green channel
cutout2=I_c[40:140, 50:250, 2] 
plt.imshow(cutout2, cmap="gray")
plt.title("Cutout 2")
plt.axis('off')
plt.show()

#Question: Why would you use different color maps?
#   Različne barvne lestvice lahko uporabimo, da bolje prikažemo in poudarimo informacije v sliki, lahko pa izboljšajo tudi
#   zaznavo kontrastov, struktur ali vzorcev, ki bi bili v RGB ali grayscale sliki manj vidni.
#   Izbira colormap-a je odvisna tudi od vrste podatkov in od tega, kaj želimo analizirati.

#d)
I_d=np.copy(I)
I_d[120:240, 200:450, 0] = 1 - I_d[120:240, 200:450, 0]
I_d[120:240, 200:450, 1] = 1 - I_d[120:240, 200:450, 1]
I_d[120:240, 200:450, 2] = 1 - I_d[120:240, 200:450, 2]
plt.figure(figsize=(10, 8))
plt.imshow(I_d)
plt.title("Inverting Part Of The Image")
plt.axis('off')
plt.show()

#Question: How is inverting a grayscale value defined?
#   Inverzija grayscale vrednosti je definirana kot odštevanje vrednosti pikslov od največje možne vrednosti v območju (če imamo uint8 torej 255 - pxl, pri float64 pa 1 - pxl).
#   Inverzija v bistvu ustvari "fotografski negativ" slike sivine, kjer temna območja postanejo svetla, svetla pa temna.


#e)
I = imread_gray("./images/umbrellas.jpg")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(I, vmin=0, vmax=1, cmap="gray")
plt.title("Original Image")
plt.axis('off')

I_e=np.copy(I)
#Debug
#print(I_e[1:5, 1:5])
big = I_e.max()
scale = big / (0.3)
I_e /= scale
#Debug
#print("big:", big, "    scale:", scale, '\n', I_newe[1:5, 1:5])

plt.subplot(1, 2, 2)
plt.imshow(I_e, vmin=0, vmax=1, cmap="gray")
plt.title("Reduced Image")
plt.axis('off')

plt.show()


#----------------------------------------------
#EXERCISE 2
#a)
threshold = 0.3
I = imread_gray("./images/bird.jpg")
height, width = I.shape

I_mask = np.zeros((height, width), dtype=np.float64)
I_mask = I >= threshold
#second option
#I_mask = np.where(I >= threshold, 1, 0)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(I, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(I_mask, cmap="gray")
plt.title("Binary Mask")
plt.axis('off')

plt.show()


#b)
def myhist(I_gray, n_bins):
    H = np.zeros(n_bins)
    I_1D = I_gray.reshape(-1)

    for i in range (I_1D.size):
        bin_index = int(I_1D[i] * n_bins)
        if bin_index == n_bins:  # To se zgodi ob I_1D[i] == 1.0 -> patch za max vrednost
            bin_index = n_bins - 1
        H[bin_index] += 1

    H /= I_1D.size
    return H

I_b1=np.copy(I)
I_b2=np.copy(I)
hist1 = myhist(I_b1, 100)
hist2 = myhist(I_b2, 20)

plt.figure(figsize=(15, 8))

plt.subplot(1, 3, 1)
plt.imshow(I, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.bar(range(len(hist1)), hist1)
plt.title("Histogram 100 bins")

plt.subplot(1, 3, 3)
plt.bar(range(len(hist2)), hist2)
plt.title("Histogram 20 bins")

plt.tight_layout()
plt.show()

#Question: The histograms are usually normalized by dividing the result by the sum of all cells. Why is that?
#   Normalizacija zagotavlja, da histogram opisuje, kako so intenzitete porazdeljene, ne pa samo koliko pikslov je (dobimo verjetnostno porazdelitev).
#   To pomeni, da vsak bin predstavlja relativno frekvenco intenzitet, ne samo število pikslov. Vsota vseh binov je ena.
#   To omogoča primerjavo med slikami različnih velikosti, saj imajo različne slike lahko različno število pikslov.


#c)
n = 50
y_limit = 0.2

I1 = imread("./images/2025-10-16_22-28-47-100.jpg")
I1_gray = imread_gray("./images/2025-10-16_22-28-47-100.jpg")
hist1 = myhist(I1_gray, n)
I2 = imread("./images/2025-10-16_22-28-47-200.jpg")
I2_gray = imread_gray("./images/2025-10-16_22-28-47-200.jpg")
hist2 = myhist(I2_gray, n)
I3 = imread("./images/2025-10-16_22-28-47-300.jpg")
I3_gray = imread_gray("./images/2025-10-16_22-28-47-300.jpg")
hist3 = myhist(I3_gray, n)
I4 = imread("./images/2025-10-16_22-28-47-400.jpg")
I4_gray = imread_gray("./images/2025-10-16_22-28-47-400.jpg")
hist4 = myhist(I4_gray, n)
I5 = imread("./images/2025-10-16_22-28-47-500.jpg")
I5_gray = imread_gray("./images/2025-10-16_22-28-47-500.jpg")
hist5 = myhist(I5_gray, n)

plt.figure(figsize=(20, 10))

plt.subplot(2, 5, 1)
plt.imshow(I1)
plt.axis('off')
plt.subplot(2, 5, 6)
plt.bar(range(len(hist1)), hist1)
plt.ylim(0, y_limit)

plt.subplot(2, 5, 2)
plt.imshow(I2)
plt.axis('off')
plt.subplot(2, 5, 7)
plt.bar(range(len(hist2)), hist2)
plt.ylim(0, y_limit)

plt.subplot(2, 5, 3)
plt.imshow(I3)
plt.axis('off')
plt.subplot(2, 5, 8)
plt.bar(range(len(hist3)), hist3)
plt.ylim(0, y_limit)

plt.subplot(2, 5, 4)
plt.imshow(I4)
plt.axis('off')
plt.subplot(2, 5, 9)
plt.bar(range(len(hist4)), hist4)
plt.ylim(0, y_limit)

plt.subplot(2, 5, 5)
plt.imshow(I5)
plt.axis('off')
plt.subplot(2, 5, 10)
plt.bar(range(len(hist5)), hist5)
plt.ylim(0, y_limit)

plt.tight_layout()
plt.show()

#Narejenih je 5 slik, vsaka z malo drugačno osvetlitvijo. Pri temnejših vidimo, da je večji del histograma na levi, torej bližje 0, saj je več pikslov z manj osvetlitve.
#Pri svetlejših obratno vidimo dokaj veliko svetlejših pikslov (bolj belih), zato je histogram bolj poln tudi na desni. Zanimivo, da je pri prvi sliki največ pikslov nekje
#sredi histograma -> gre namreč za  to, da čeprav je predmet osvetljen ni to popolnoma prava bela barva (grayscale ki smo ga naredili je povprečje vseh treh signalov RGB, kar 
#pomeni, da je verjetno listek rahlo rumenkaste, oker barve)


#d)
def otsu_threshold(I_gray, n_bins):
    hist = myhist(I_gray, n_bins)
    bins = np.arange(n_bins)

    val_max = -1
    thr = -1

    for t in range(1, n_bins):
        # vsote
        n1 = np.sum(hist[:t])
        n2 = np.sum(hist[t:])
        if n1 == 0 or n2 == 0:
            continue

        # means
        m1 = np.sum(bins[:t] * hist[:t]) / n1
        m2 = np.sum(bins[t:] * hist[t:]) / n2

        # between-class variance
        val = n1 * n2 * (m1 - m2) ** 2

        # max
        if val > val_max:
            val_max = val
            thr = t

    # normalize threshold to [0, 1]
    threshold_value = thr / n_bins
    return threshold_value

I_gray = imread_gray("./images/bird.jpg")
t = otsu_threshold(I_gray, 256)
I_mask = I_gray >= t

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(I_gray, cmap='gray')
plt.title("Otsu")
plt.axis('off')

plt.subplot(1, 3, 2)
hist = myhist(I_gray, 256)
plt.plot(hist)
plt.axvline(x=t*256, color='r', linestyle='--', label=f"T={t:.3f}")
plt.legend()
plt.title("Histogram + Otsu threshold")

plt.subplot(1, 3, 3)
plt.imshow(I_mask, cmap='gray')
plt.title("Binarized Image")
plt.axis('off')

plt.tight_layout()
plt.show()


#----------------------------------------------
#EXERCISE 3
#a)
I = imread("./images/mask.png")
n = 5
SE = np.ones((n,n)) # create a square structuring element
I_eroded = cv2.erode(I, SE)

n = 10
SE = np.ones((n,n))
I_dilated = cv2.dilate(I, SE)

def opening(img, SE):
    I_mix1 = cv2.erode(img, SE)
    I_mix2 = cv2.dilate(I_mix1, SE)
    return I_mix2

def closing(img, SE):
    I_mix3 = cv2.dilate(img, SE)
    I_mix4 = cv2.erode(I_mix3, SE)
    return I_mix4

n = 7
SE = np.ones((n,n)) # create a square structuring element
I_close = closing(I, SE)
I_open = opening(I, SE)

plt.figure(figsize=(12, 5))

plt.subplot(1, 4, 1)
plt.imshow(I_eroded, cmap="gray")
plt.title("Erosion")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(I_dilated, cmap="gray")
plt.title("Dilation")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(I_open, cmap="gray")
plt.title("Opening")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(I_close, cmap="gray")
plt.title("Closing")
plt.axis('off')

plt.show()

#Question: Based on the results, which order of erosion and dilation operations produces opening and which closing?
#   opening: erode -> dilate
#   closing: dilate -> erode

#b)
I_gray = imread_gray("./images/bird.jpg")
t = otsu_threshold(I_gray, 256)
I_mask = I_gray >= t
I_mask_float = I_mask.astype(np.float64)
n = 25
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
I_mask_close = closing(I_mask_float, SE)

#n = 5
#SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
#I_close = closing(I_mask_float, SE)

plt.figure(figsize=(12, 5))
plt.imshow(I_mask_close, cmap="gray")
plt.title("Bird Mask")
plt.axis('off')
plt.show()

#c)
def immask(I, mask):
    mask = np.expand_dims(mask, axis=2)
    return I * mask

I = imread("./images/bird.jpg")
I_out = immask(I, I_mask_close)

plt.figure(figsize=(12, 5))
plt.imshow(I_out)
plt.title("Bird Without Background")
plt.axis('off')
plt.show()

#d)
I_gray = imread_gray("./images/eagle.jpg")
t = otsu_threshold(I_gray, 256)
I_mask = I_gray < t

I = imread("./images/eagle.jpg")

I_out = immask(I, I_mask)
plt.figure(figsize=(12, 5))
plt.imshow(I_out)
plt.title("Eagle Without Background")
plt.axis('off')
plt.show()

#Question: Why is the background included in the mask and not the object? How would you fix that in general? (just inverting the mask if necessary doesn’t count)
#   Ker je ozadje svetlejše od objekta, bodo piksli ozadja nad pragom -> v maski = 1, medtem ko je objekt = 0
#   Na splošno to lahko to popravimo s predhodno obdelavo slike (Normalizacija, izboljšanje kontrasta) ali pa uporabimo kakšne robne oz. teksturne
#   teksturne značilke (Namesto same intenzitete uporabimo gradient, robove, teksturo ali barvo, ki razlikuje objekt od ozadja)
#   Glavna ideja je torej, da se ne zanašamo samo na intenziteto, ampak uporabimo dodatne informacije o objektu.

#e)
I = imread("./images/coins.jpg")
height, width, channels = I.shape
I_gray = imread_gray("./images/coins.jpg")
t = otsu_threshold(I_gray, 256)
I_mask = (I_gray < t).astype(np.uint8) * 255 #connectedComponent potrebuje masko/sliko v foramtu uint8, zato pretvorimo
n = 30
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
I_mask_close = closing(I_mask, SE)

(number, labels, stats, centers) = cv2.connectedComponentsWithStats(I_mask_close)

I_mask_clean = np.copy(I_mask_close)

for i in range (1, number):
    if stats[i, cv2.CC_STAT_AREA] > 700:
        I_mask_clean[labels == i] = 0


I_white = np.ones((height, width), dtype=np.float64)
I_white = np.expand_dims(I_white, axis=2)
#masko moram delit z 255, saj je v formatu uint8, I_white pa v float64
I_out = immask(I_white, I_mask_clean // 255)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(I)
plt.title("Original Coin Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(I_out, cmap='gray')
plt.title("Mask Without Large Coins")
plt.axis('off')
plt.show()