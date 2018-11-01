import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as f


#abre la imagen la convierte en una matrix de flotantes
Arboles = plt.imread('Arboles.png').astype(float)
#de no estar los valores entre 0 y 1 los normaliza
Arboles = Arboles/np.amax(Arboles)
#calcula la transformada
ArbolesFFT =f.fft2(Arboles)
#centra la transformada
ArbolesFFT2=f.fftshift(ArbolesFFT)
#calcula el valor de la transformad y luego la centra
plt.imshow(f.fftshift(abs(ArbolesFFT)),cmap = 'gray')
plt.title("Transformada de Fourier")
plt.savefig("GualdronTonny_FT2D.pdf")
plt.close()

# de la grafica anterior se notan que ha cuatro puntos que son mas tenues que el central, es decir que el central correponde al la imagen y los otros al ruido entonces deberiamos volver esos puntos ceros
#numero de lineas
l=np.shape(Arboles)[0]
#numero de columnas
c=np.shape(Arboles)[1]
#Crea una matriz del mismo tamaño con solo unos
frecuencias = np.ones((l,c)).astype(complex)
#Busca las 5 posiciones mas brillantes en orden de intensidad
copia=f.fftshift(abs(ArbolesFFT))
ii=[]
jj=[]
ia=0
ja=0
m=copia[0][0]
for z in range(5):    
    for i in range(l):
        for j in range(c):
            if(copia[i,j]>m):
                m=copia[i,j]
                ia=i
                ja=j
    copia[ia,ja]=0
    ii.append(ia)
    jj.append(ja)
    ia=0
    ja=0
    m=copia[0][0]

#convierte los 4 puntos mas brillantes despues del maximo en 0
frecuencias[ii[1],jj[1]]=0
frecuencias[ii[2],jj[2]]=0
frecuencias[ii[3],jj[3]]=0
frecuencias[ii[4],jj[4]]=0
#Crea una matriz donde se guardara el resultado
nueva = np.ones((l,c)).astype(complex)
#Se multiplica uno a uno para volver solo esos cuatro puntos 0
for i in range(l):
    for j in range(c):
         nueva[i,j]=ArbolesFFT2[i,j]*frecuencias[i,j]
#Halla de nuevo la imagen
nueva2=f.ifft2(f.ifftshift(nueva))
plt.imshow(abs(nueva),cmap = 'gray')
plt.title("Transformada de Fourier Filtrada")
plt.savefig("GualdronTonny_FT2D_filtrada.pdf")
plt.close()

#verifica que las posiciones de los 4 puntos ruidos sean 0
if(nueva[ii[1],jj[1]]==0 and nueva[ii[2],jj[2]]==0 and nueva[ii[3],jj[3]]==0 and nueva[ii[4],jj[4]]==0):
    plt.imshow(abs(nueva2),cmap = 'gray')
    plt.title("Imagen Final")
    plt.savefig("GualdronTonny_Imagen_filtrada.pdf")
    plt.close()
else:
    print("No se pudo eliminar el ruido") 
