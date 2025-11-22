import a2_utils as a2
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

signal_c = a2.read_data("c.txt")
kernel_a = a2.read_data("kernel.txt")
kernel_b = a2.read_data("b.txt")

convolve_a = complex_convolution(signal_c, kernel_a)
convolve_b = complex_convolution(signal_c, kernel_b)



plt.figure(figsize=(15, 8))
plt.suptitle("Convolution")
plt.subplot(2, 3, 1)
plt.plot(signal_c)
plt.title("Original Signal")
plt.subplot(2, 3, 2)
plt.plot(kernel_a)
plt.title("Kernel a")
plt.subplot(2, 3, 3)
plt.plot(kernel_b)
plt.title("Kernel b")
plt.subplot(2, 3, 5)
plt.plot(convolve_a)
plt.subplot(2, 3, 6)
plt.plot(convolve_b)

plt.show()