import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from scipy import fftpack as f
#abre la imagen la convierte en una matrix de flotantes
Arboles = plt.imread('Arboles.png').astype(float)

ArbolesFFT = f.fft2(Arboles)
ArbolesFFTNorma= np.abs(ArbolesFFT)
minA= np.amin(np.abs(ArbolesFFT))
maxA= np.amax(np.abs(ArbolesFFT))
plt.imshow(ArbolesFFTNorma, norm=pltc.LogNorm(vmin=minA,vmax=maxA))
plt.colorbar()
plt.show()

k=0.1
CopiaArboles=ArbolesFFT.copy()
l=np.shape(CopiaArboles)[0]
c=np.shape(CopiaArboles)[1]

CopiaArboles[int(l*k):int(l*(1-k))]=1
CopiaArboles[:,int(c*k):int(c*(1-k))]=1

ArbolesFFTNorma2= np.abs(CopiaArboles)
plt.imshow(ArbolesFFTNorma2, norm=pltc.LogNorm(vmin=minA,vmax=maxA))
plt.colorbar()
plt.show()

ArbolesFil = f.ifft2(CopiaArboles).real

plt.figure()
plt.imshow(ArbolesFil, plt.cm.gray)
plt.title('Reconstructed Image')
plt.show()
