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

#datos para la validacion
x_val=np.arange(-10,10,.5)
x_val=x_val.reshape(-1,1)
y_val=a*x_val**2+b*x_val+c-.1*x_val**3
z_val=y_val+np.random.normal(0, sigma,y_val.shape)

#Representa graficamente los datos
plt.figure(1)
lines = plt.plot(x, z)
plt.setp(lines, color='b', linewidth=2.0)
lines1 = plt.plot(x_val, z_val)
plt.setp(lines1, color='g', linewidth=2.0)
plt.title("Datos a estimar")
plt.show()


#######################
#Entrena
#######################

#Regulariza los datos
x_corr=max(abs(x))
z_corr=max(abs(z))
x_val_corr=max(abs(x_val))
z_val_corr=max(abs(z_val))

plt.figure(2)
lines = plt.plot(x/x_corr, z/z_corr)
plt.setp(lines, color='b', linewidth=2.0)
lines1 = plt.plot(x_val/x_val_corr, z_val/z_val_corr)
plt.setp(lines1, color='g', linewidth=2.0)
plt.title("Datos a estimar (escala)")
plt.show()

#Prepara el dataset
tot=len(x)
tam_muestra=10
ciclos_entrenamiento=150
train_dataset=tf.data.Dataset.from_tensor_slices(((x/x_corr).astype(np.float32),
                                                  (z/z_corr).astype(np.float32))).shuffle(tot).batch(tam_muestra)

train_dataset_val=tf.data.Dataset.from_tensor_slices(((x_val/x_val_corr).astype(np.float32),
                                                  (z_val/z_val_corr).astype(np.float32))).shuffle(tot).batch(tam_muestra)

#Callbacks
#Callbacks. Tensorboard
tensorboard_cbk=keras.callbacks.TensorBoard(
  log_dir='.\\entrena',
  histogram_freq=1, 
  update_freq='epoch', # indica que se generaran muestras de datos por cada epoch
#  update_freq='batch', # indica que se generaran muestras de datos por cada batch
  profile_batch = 0)

#Callbacks. Parada anticipada del entrenamiento
paradaTemprana_cbk=keras.callbacks.EarlyStopping(
        # Observa la funcion de perdida - tipicamente usariamos val_loss en lugar de loss
        monitor='val_loss',
        # Si la funcion de error cambia menos de min_delta
        min_delta=1e-3,
        # en patience batchs, para
        patience=6,
        verbose=1)

#Tasa de aprendizaje variable
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#Loop de entrenamiento
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.mean_squared_error,tf.keras.metrics.Accuracy()])

modelo.fit(train_dataset,validation_data=train_dataset_val, epochs=ciclos_entrenamiento,callbacks=[paradaTemprana_cbk,tensorboard_cbk])

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