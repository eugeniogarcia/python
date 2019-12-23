from tensorflow import keras
from tensorflow.keras import layers,Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#######################
#Define el modelo
#######################
class Cuadratica(layers.Layer):

  def __init__(self):
    super(Cuadratica, self).__init__()
   
    
  def build(self, input_shape):
    self.a = self.add_weight(shape=(input_shape[-1],1),
                             initializer='random_normal',
                             dtype=tf.float32,
                             trainable=True,
                             name="a")
    self.b = self.add_weight(shape=(input_shape[-1],1),
                             initializer='random_normal',
                             dtype=tf.float32,
                             trainable=True,
                             name="b")
    self.c = self.add_weight(shape=1,
                             dtype=tf.float32,
                             initializer='zeros',
                             trainable=True,
                             name="c")

  def call(self, inputs):
    x=((inputs**2) * self.a) + (inputs * self.b) + self.c
    return x

modelo=Sequential()
modelo.add(Cuadratica())

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

train_dataset=tf.data.Dataset.from_tensor_slices(((x/x_corr).astype(np.float32),
                                                  (z/z_corr).astype(np.float32))).repeat().shuffle(tot).batch(tam_muestra)


#loop de entrenamiento

optimizer =tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn=tf.keras.losses.MeanSquaredError()

epochs=200
for x_batch_train, y_batch_train in train_dataset.take(int(epochs*tot/tam_muestra)):
  with tf.GradientTape() as tape:
    resp_est = modelo(x_batch_train)
    error = loss_fn(y_batch_train, resp_est)
    error+= sum(modelo.losses)
  
  grads = tape.gradient(error, modelo.trainable_weights)
  optimizer.apply_gradients(zip(grads, modelo.trainable_weights))


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


print(np.mean(abs(y_est-z))/tot)
print("a={}".format(modelo.weights[0]/(x_corr**2)*z_corr))
print("b={}".format(modelo.weights[1]/x_corr*z_corr))
print("c={}".format(modelo.weights[2]*z_corr))