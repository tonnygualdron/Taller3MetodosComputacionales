import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as f
from scipy.stats import mode


#abre la imagen la convierte en una matrix de flotantes
Arboles = plt.imread('Arboles.png').astype(float)
#calcula la transformada
ArbolesFFT = f.fft2(Arboles)
#la centra
ArbolesFFT2 = f.fftshift(ArbolesFFT)
#Calcula las magnitudes y saca logaritmo para lograr una escala mejor
NormaFFT = np.log(np.abs(ArbolesFFT2))
#Genera la grafica con las magnitudes centradas de la transformada en escala de grises
plt.imshow(NormaFFT, cmap = 'gray')
plt.colorbar()
plt.title("Transformada de Fourier")
plt.savefig("GualdronTonny_FT2D.pdf")
plt.close()

ArbolesFFT4=np.ones((l)).astype(complex)
for i in range(np.shape(ArbolesFFT)[0]):
     ArbolesFFT3[i]=ArbolesFFT[i,i]

moda=mode(ArbolesFFT)
l=np.shape(ArbolesFFT)[0]
c=np.shape(ArbolesFFT)[1]
ArbolesFFT3=np.ones((l,c)).astype(complex)
for j in range(np.shape(ArbolesFFT)[1]):
    for i in range(np.shape(ArbolesFFT)[0]):
          ArbolesFFT3[i,j]=ArbolesFFT[i,j]+moda[0][0][j]
ArbolesFil=f.ifft2(ArbolesFFT3).real
plt.imshow(ArbolesFil, cmap = 'gray')
plt.show()
ArbolesFinal=Arboles-ArbolesFil
plt.imshow(ArbolesFinal, cmap = 'gray')
plt.show()

