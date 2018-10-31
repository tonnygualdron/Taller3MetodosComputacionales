import numpy as np
import numpy.linalg as n
import matplotlib.pyplot as plt

# Abre el documento WDBC.dat.
archivo = open('WDBC.dat','r')
# Lee todas las lineas
lineas = archivo.readlines()
# cuenta las lineas
tamano = len(lineas)

# llama linea por linea la divide por separador coma y corrige que en el ultimo se agarega un \n
datos = []

for i in range(tamano):
    actual=lineas[i].split(',')
    correcion= actual[31].split('\n')
    actual.pop(31)
    actual.append(correcion[0])
    datos.append(actual)

# los datos estan ordenados asi
# 0) ID number
# 1) Diagnosis (M = malignant, B = benign)
# 2-31)
#	2) radius 
#	3) texture (standard deviation of gray-scale values)
#	4) perimeter
#	5) area
#	6) smoothness (local variation in radius lengths)
#	7) compactness (perimeter^2 / area - 1.0)
#	8) concavity (severity of concave portions of the contour)
#	9) concave points (number of concave portions of the    #contour)
#	10) symmetry 
#	11) fractal dimension ("coastline approximation" - 1)

# remplazaresmos maligno por 1 y benign por 0

for i in range(tamano):
    if (datos[i][1] == 'M'):
        datos[i][1] = '1'
    if (datos[i][1] == 'B'):
        datos[i][1] = 0

# debido a que el numero de ID no tiene relacion con las varialbles medidas entonces crearemos una matriz que retira ID y tambien el M o B, debido a que este tampoco es una variable.
datos2=[]
datos1=[]
#datos1 guarda M o B para cada paciente
for i in range(tamano):
    actual=datos[i]
    datos1.append(actual[1])
    actual.pop(0)
    actual.pop(0)
    datos2.append(actual)
#convierte las matrices a numpy array para facilidad de procesamiento
datos1=np.asarray(datos1)
datos1=datos1.astype(np.float)
datos3=np.asarray(datos2)
datos3=datos3.astype(np.float)

#Implementacion de lo vistos en clase y de la pagina web https://relopezbriega.github.io/blog/2015/06/27/probabilidad-y-estadistica-con-python/#%C2%BFQu%C3%A9-es-la-Estad%C3%ADstica?
filas = np.shape(datos3)[0]
columnas = np.shape(datos3)[1]
#normaliza los datos
datosNon = np.zeros([filas, columnas])
for i in range(filas):
    u1 = np.mean(datos3[i])
    s1 = np.std(datos3[i])
    datosNon[i] = (datos3[i]-u1)/s1
# Crea una matriz de ceros, cada cero luego sera remplazado por el valor de la covarianza
cov = np.zeros([columnas, columnas])
for i in range(columnas):
    for j in range(columnas):
        u1 = np.mean(datosNon[:,i])
        u2 = np.mean(datosNon[:,j])
        cov[i,j] = np.sum((datosNon[:,i]-u1) * (datosNon[:,j]-u2)) / (filas-1)

aut=n.eig(cov)
autvec=aut[1]
autval=aut[0]
#Imprime los auto valores con su correpondiente autovector.
for i in range(30):
   print("El autovalor igual a ", autval[i], " corresponde a vector ", autvec[i])
#suma las magnitudes de los vectores
sumaTotal=np.sum(autval)
pval=np.zeros([len(autval)])
#convierte cada auto valor en un porcentaje
for i in range(len(autval)):
    pval[i]=(autval[i]/sumaTotal)*100
#retorna el numero de variables 

x=datosNon[:,0]
y=datosNon[:,1]
vec=np.array([autvec[0],autvec[1]])

nueva=np.dot(datosNon,np.transpose(vec))

x=nueva[:,0]
y=nueva[:,1]
z=datos1

x0=[]
y0=[]
x1=[]
y1=[]
for i in range(len(x)):
    if(z[i]==1):
        x1.append(x[i])
        y1.append(y[i])
    if(z[i]==0):
        x0.append(x[i])
        y0.append(y[i])  

plt.scatter(x0,y0,label='B',alpha=0.3)
plt.scatter(x1,y1,label='M',alpha=0.3)
plt.show()








