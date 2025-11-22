import cv2
import numpy as np
from matplotlib import pyplot as plt

a = (1, 2, 3)
#touples doesnt support reassignment
#a(0) = 5 .... ne gre, error dobiš

len(a)
print(a)
for item in a:
    print(item)

if 3>5:
    print("True")
else:
    print("False")

for idx, item in enumerate([1, 100, 1000]):
    print(idx, item)



a = [1, 2, 3, 4, 5]
a[-2]
a[3:]
a = np.arange(10)
print(a)

for i in range(5, 10, 2):
    print(i)


[item**2 for item in a]

a = np.array([[1, 2], [3, 4]])
a.shape

img = cv2.imread('./assigment1/images/bird.jpg')
img.shape
plt.imshow(img)
img_gray = np.mean(img, axis=2)
plt.imshow(img_gray, cmap = 'gray')
plt.show()

mask = img_gray > 40
mask.shape

np.max(mask)
np.min(mask)
masked_image = img_gray * mask
plt.imshow(masked_image)

#Overflow, ker seštevamo v uint8 le do 255(ne gre več) -> narobe
img_gray = img[:,:,0] + img[:,:,1] + img[:,:,2] / 3


N = 10
im = np.zeros((N,N))
x = 1
y = 2
im[x,y] = 1
plt.clf()
plt.imshow(im)
#višina in širina v mathplotlib sta zamenjana
plt.plot(x,y,'r.')
#zato samo zamenjaj koordinate in bo slika prava
plt.show()


#mathplotlib sam prilagaja največjo in najmanjšo vrednost, 
#zato moraš sam dodati vmin in vmax, da vidiš razliko, če jo iščeš
N = 5
x = y = N//2
im1 = np.zeros((N,N), dtype=np.uint8)
im2 = np.zeros((N,N), dtype=np.uint8)
im3 = np.zeros((N,N), dtype=np.float32)
im4 = np.zeros((N,N), dtype=np.float32)

#želimo imeti v enakem merilu
im1[x, y]=1
im2[x, y]=255
im3[x, y]=1e-3
im4[x, y]=1e3
plt.subplot(2,2,1)
plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
plt.subplot(2,2,2)
plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
plt.subplot(2,2,3)
plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
plt.subplot(2,2,4)
plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
plt.show()