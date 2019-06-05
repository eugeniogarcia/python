#Importa los modulos
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter  # useful for `logit` scale


x=np.arange(-20,20,.2)
y1=np.square(x-3)
y2=np.square(x+3)

#Podemos especificar las propiedades de forma progresiva
plt.figure(1)
lines = plt.plot(x, y1, x, y2)

# usando key=valor
plt.setp(lines, color='r', linewidth=2.0)
# estilo MATLAB
plt.setp(lines, 'color', 'b', 'linewidth', 2.0)

#lines.set_antialiased(False)
plt.show()

plt.figure(2)
#Genera la variable independiente en intervalos de 0.2
t = np.arange(0., 5., 0.2)

#- rojos
#s , cuadrados, azules
#^ , triangulos, verdes
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

# Define una funcion
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

#Genera la variable independiente
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

#Crea la figura uno
plt.figure(3)
plt.title="Representacion de una funcion"
plt.subplot(211)
plt.plot(t1, f(t1), 'bo',t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()  

# Fija el seed para el generador de numeros aleatorios
np.random.seed(19680801)

#Parametros de la normal
mu, sigma = 100, 15
#(x-mu)/sigma sigue una normal 0,1. Generamos 10000 puntos
x = mu + sigma * np.random.randn(10000)

# representamos un histograma
plt.figure(4)

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('Inteligencia')
plt.ylabel('Probabilidad')
plt.title('Histograma del IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

#Poner anotaciones en el grafico
plt.figure(5)

ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

plt.ylim(-2,2)
plt.show()

#Escala logaritmica
plt.figure(6)

# normal de media 0.5 y desviacion 0.4
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
#Nos quedamos con aquellos valores que estan entre 0 y 1. El shape
#del vector resultante sera diferente
y = y[(y > 0) & (y < 1)]

y.sort()

x = np.arange(len(y))

# plot with various axes scales
plt.figure(7)

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)


# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())

# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()