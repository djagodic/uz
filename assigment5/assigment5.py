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
print(d)


#c)
f = 2.5
T = 120
pixelWidth = 7.4e-3
x1 = 550
x2 = 300
pz = (T * f) / ((x1 - x2)*pixelWidth)
print("Razdalja:", pz)

