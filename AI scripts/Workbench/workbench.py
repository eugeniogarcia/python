from tensorflow import keras
from tensorflow.keras import layers,Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#######################
#Define el modelo
#######################

modelo=Sequential()
modelo.add(layers.Dense(10, activation='sigmoid', name='dense_1', input_shape=(1,)))
modelo.add(layers.Dense(10, activation='sigmoid', name='dense_2'))
modelo.add(layers.Dense(1, activation='sigmoid', name='dense_3'))

#######################
#Prepara los datos
#######################

#ruido
sigma=10

#modelo
a=1
b=-3
c=5

#datos
x=np.arange(-10,10,.1)
x=x.reshape(-1,1)
y=a*x**2+b*x+c-.1*x**3
z=y+np.random.normal(0, sigma,y.shape)

#Representa graficamente los datos
plt.figure(1)
lines = plt.plot(x, z)
plt.setp(lines, color='b', linewidth=2.0)
plt.title("Datos a estimar")
plt.show()


#######################
#Entrena
#######################

#Regulariza los datos
x_corr=max(abs(x))
z_corr=max(abs(z))

plt.figure(2)
lines = plt.plot(x/x_corr, z/z_corr)
plt.setp(lines, color='b', linewidth=2.0)
plt.title("Datos a estimar (escala)")
plt.show()

#Prepara el dataset
tot=len(x)
tam_muestra=10
ciclos_entrenamiento=200
train_dataset=tf.data.Dataset.from_tensor_slices(((x/x_corr).astype(np.float32),
                                                  (z/z_corr).astype(np.float32))).shuffle(tot).batch(tam_muestra)

#Loop de entrenamiento

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.mean_squared_error])

modelo.fit(train_dataset, epochs=ciclos_entrenamiento)

#######################
#Inferencia
#######################

#Estima
y_est=modelo.predict(x/x_corr)*z_corr

plt.figure(3)
plt.title("Estimacion vs Reales")
lines = plt.plot(x, y_est)
lines2 = plt.plot(x,z)
plt.setp(lines, color='g', linewidth=2.0)
plt.setp(lines2, color='b', linewidth=2.0)
plt.show()