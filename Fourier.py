import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as f
from scipy.interpolate import interp1d

#utilizare lineas de codigo de PCR.py para leer el archivo
# Abre el documento signal.dat
archivo = open('signal.dat','r')
# Lee todas las lineas
lineas = archivo.readlines()
# cuenta las lineas
tamano = len(lineas)

# llama linea por linea la divide por separador coma y corrige que en el ultimo se agarega un \n
datos = []

for i in range(tamano):
    actual=lineas[i].split(',')
    correcion= actual[1].split('\n')
    actual.pop(1)
    actual.append(correcion[0])
    datos.append(actual)

#convierte las matrices a numpy array para facilidad de procesamiento
signal=np.asarray(datos)
signal=signal.astype(np.float)

archivo.close()

# Abre el documento incompletos.dat
archivo2 = open('incompletos.dat','r')
# Lee todas las lineas
lineas2 = archivo2.readlines()
# cuenta las lineas
tamano2 = len(lineas2)

# llama linea por linea la divide por separador coma y corrige que en el ultimo se agarega un \n
datos2 = []

for i in range(tamano2):
    actual=lineas2[i].split(',')
    correcion= actual[1].split('\n')
    actual.pop(1)
    actual.append(correcion[0])
    datos2.append(actual)

#convierte las matrices a numpy array para facilidad de procesamiento
inco=np.asarray(datos2)
inco=inco.astype(np.float)

archivo2.close()

plt.plot(signal[:,0],signal[:,1])
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]") 
plt.title("Signal")
plt.savefig("GualdronTonny_signal.pdf")
plt.close()
def TDF(signal):
    N = np.shape(signal)[0]
    lista = np.ones((N)).astype(complex)
    for m in range(N):
        R = 0.0
        for n in range(N):
            R += signal[n][1] * np.exp(- 2j * np.pi * m * n / N)
        lista[m]=R
    return lista

def TDF2(signal):
    N = np.shape(signal)[0]
    lista = np.ones((N)).astype(complex)
    for m in range(N):
        R = 0.0
        for n in range(N):
            R += signal[n] * np.exp(- 2j * np.pi * m * n / N)
        lista[m]=R
    return lista

trans=TDF(signal)
x=f.fftfreq(np.shape(signal)[0],signal[1][0])
y=trans
plt.vlines(x,0,y.imag)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud") 
plt.title("Signal TF")
plt.savefig("GualdronTonny_TF.pdf")
plt.close()

y2=abs(y)
ii=[]
ia=0
m=0
for z in range(3):
    for i in range(len(x)):
        if(y2[i] > m):
            m=y2[i]
            ia=i
    ii.append(ia)
    y2[ia]=0
    ia=0
    m=0

print("Las frecuencias del sistema corresponde a ", x[ii[0]], " ,",x[ii[1]]," y ", x[ii[2]])

print("Los datos incompletos no se pueden interpolar debido a que estos no estas igualmente espaciados por lo cual una transformada discreta seria ilogico")

#Codigo reciclado de la tarea 2
f2 = interp1d(inco[:,0], inco[:,1], kind='quadratic')
f3 = interp1d(inco[:,0], inco[:,1], kind='cubic')

xx = np.linspace(min(inco[:,0]),max(inco[:,0]),512)

yq=TDF2(f2(xx))
yc=TDF2(f3(xx))
dt=(max(inco[:,0])-min(inco[:,0]))/512
xxx=f.fftfreq(512,dt)


plt.figure()

plt.subplot(3,1,1)
plt.vlines(x,0,y.imag)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud") 
plt.title("Signal TF")

plt.subplot(3,1,2)
plt.vlines(xxx,0,yq.imag)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud") 
plt.title("Cuadratica")
	
plt.subplot(3,1,3)
plt.vlines(xxx,0,yc.imag)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud") 
plt.title("Cubica")
	
plt.savefig("GualdronTonny_TF_interpola.pdf")
plt.close()

print("Las señales interpoladas presentan mas picos de amplitud considerables que la original, es decir hay un numero mayor de frecuencias que describen al sistema, la TF no estan limpia como la original. Esto puede ser resultado de que se propaga el error en el metodo de aproximación de la interpolación.")

