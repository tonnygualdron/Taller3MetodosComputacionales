import numpy as np
import matplotlib.pyplot as plt
import cmath as cp

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

archivo.close()

#plt.plot(signal[:,0],signal[:,1])
#plt.xlabel("Tiempo [s]")
#plt.ylabel("Voltaje [V]") 
#plt.title("Signal")
#plt.savefig("GualdronTonny_signal.pdf")

def TransDisFourier(signal,k):
    n = np.shape(signal)[0] # numero de puntos igual
    x=signal[:,1]
    t=signal[:,0]
    dt=signal[1][0]
    R=0
    R=x*(np.exp(((-2j*np.pi*t*k)/(n))))
    Re=[]
    te=[]
    for i in range(len(R)):
        ac=(R[i].real**2 + R[i].imag**2)**0.5
        Re.append(ac)
    for i in range(len(t)):
        ac=i/dt
        te.append(ac)
    return (te,Re)


trans=TransDisFourier(signal,1)
plt.plot(trans[0],trans[1])
plt.show()

