import numpy as np
from matplotlib import pyplot as plt
import cv2


img = cv2.imread('Arboles.png',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitudFFT = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagen de entrada'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitudFFT, cmap = 'gray')
plt.title('FFT'), plt.xticks([]), plt.yticks([])
plt.show()
