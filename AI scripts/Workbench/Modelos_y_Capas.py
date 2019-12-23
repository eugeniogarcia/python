from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#############################################
#Crea un modelo con la api funcional
#############################################
def get_uncompiled_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model

#Podemos especificar las metricas y la funcion de error especificando el tipo o el nombre
def get_compiled_model_v1():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
  return model

def get_compiled_model_v2():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
  return model

def get_compiled_model_v3():
  model = get_uncompiled_model()
  model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.MeanSquaredError())
  return model

#############################################
#Capa personalizada
#############################################
class Cuadratica_v1(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Cuadratica_v1, self).__init__()
    a_init = tf.random_normal_initializer()
    self.a = tf.Variable(initial_value=a_init(shape=(input_dim, units),
                         dtype='float32'),
                         trainable=True,
                         name="c")
    b_init = tf.random_normal_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(input_dim, units),
                         dtype='float32'),
                         trainable=True,
                         name="b")
    c_init = tf.zeros_initializer()
    self.c = tf.Variable(initial_value=c_init(shape=(units,),
                         dtype='float32'),
                         trainable=True,
                         name="c")

  def call(self, inputs):
    return tf.matmul(tf.math.square(inputs), self.a) + tf.matmul(inputs, self.b) + self.c


class Cuadratica_v2(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Cuadratica_v2, self).__init__()
    self.a = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True,
                             name="a")
    self.b = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True,
                             name="b")
    self.c = self.add_weight(shape=(units,),
                             initializer='zeros',
                             trainable=True,
                             name="c")

  def call(self, inputs):
    return tf.matmul(tf.math.square(inputs), self.a) + tf.matmul(inputs, self.b) + self.c

print("Modelo cuadratico v2")
#8 elementos de entrada, de dimension 2
x = tf.ones((8, 2))

capa_cuadratica= Cuadratica_v2(4, 2)
y = capa_cuadratica(x)
print(capa_cuadratica.trainable_weights)

#8 elementos de salida de dimension 4
print(y.shape)

#############################################
#Capa personalizada
#Variables no trainable
#############################################
#Demuestra el uso de variables no trainables
class CalculaSuma(layers.Layer):

  def __init__(self, input_dim):
    super(CalculaSuma, self).__init__()
    self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                             trainable=False)

  def call(self, inputs):
    self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    return self.total

print("Modelo CalculaSuma")
print("x")
x = tf.ones((4, 2))
print(x.numpy())
my_sum = CalculaSuma(2)

print(my_sum.trainable_weights)
print("y")
y = my_sum(x)
print(y.numpy())
print("y segunda llamada")
y = my_sum(x)
print(y.numpy())


#############################################
#Capa personalizada
#Diferir la creacion al build
#############################################

class Cuadratica_v3(layers.Layer):

  def __init__(self, units=32):
    super(Cuadratica_v3, self).__init__()
    self.units = units
    
    
  def build(self, input_shape):
    self.a = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True,
                             name="a")
    self.b = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True,
                             name="b")
    self.c = self.add_weight(shape=(self.units,),
                             initializer='zeros',
                             trainable=True,
                             name="c")

  def call(self, inputs):
    return tf.matmul(tf.math.square(inputs), self.a) + tf.matmul(inputs, self.b) + self.c

print("Modelo cuadratico v3")
#8 elementos de entrada, de dimension 2
x = tf.ones((8, 2))

capa_cuadratica= Cuadratica_v3(4)
y = capa_cuadratica(x)
print(capa_cuadratica.trainable_weights)

#8 elementos de salida de dimension 4
print(y.shape)


#############################################
#Capa personalizada
#Combinar capas
#############################################

class Lineal_v1(layers.Layer):

  def __init__(self, units=32):
    super(Lineal_v1, self).__init__()
    self.units = units
    
    
  def build(self, input_shape):
    self.a = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True,
                             name="a")
    self.b = self.add_weight(shape=(self.units,),
                             initializer='zeros',
                             trainable=True,
                             name="b")

  def call(self, inputs):
    return tf.matmul(inputs, self.a) + self.b

class Cuadratica_v4(layers.Layer):

  def __init__(self, units=32):
    super(Cuadratica_v4, self).__init__()
    self.capa1=Lineal_v1(units)  
    self.capa2=Lineal_v1(units)  

  def call(self, inputs):
    cap1=self.capa1(inputs)
    x = tf.nn.relu(cap1)
    cap2=self.capa2(x)
    return cap2

print("Modelo cuadratico v4")
#8 elementos de entrada, de dimension 2
x = tf.ones((8, 2))

capa_cuadratica= Cuadratica_v4(4)
y = capa_cuadratica(x)
print(capa_cuadratica.weights)
print(capa_cuadratica.trainable_weights)

#8 elementos de salida de dimension 4
print(y.shape)


#############################################
#Capa personalizada
#Errores
#############################################


class ErroresRegularizacion(layers.Layer):

  def __init__(self, tasa=1e-2):
    super(ErroresRegularizacion, self).__init__()
    self.tasa = tasa

  def call(self, inputs):
    self.add_loss(self.tasa * tf.reduce_sum(inputs))
    return inputs

class CapaPadre(layers.Layer):

  def __init__(self):
    super(CapaPadre, self).__init__()
    self.activity_reg = ErroresRegularizacion(1e-2)

  def call(self, inputs):
    return self.activity_reg(inputs)

print("Capa con errores")
layer = CapaPadre()
assert len(layer.losses) == 0  # La primera vez los errores deben ser cero
print("primera ejecucion")
_ = layer(tf.ones(1, 1))
assert len(layer.losses) == 1  
print(layer.losses[0].numpy()) # 1 * 1e-2
print("segunda ejecucion")
_ = layer(tf.ones(1, 1))
assert len(layer.losses) == 1  
print(layer.losses[0].numpy()) # 1 * 1e-2


#############################################
#Modelo personalizado
#Errores
#############################################

class miModelo(tf.keras.Model):
  """Un modelo personalizado de ejemplo."""

  def __init__(self,
               units=32,
               name='mi-modelo',
               **kwargs):
    super(miModelo, self).__init__(name=name, **kwargs)
    self.capa1 = Cuadratica_v4(units)
    self.capa2 = Lineal_v1(1)

  def call(self, inputs):
    x=self.capa1(inputs)
    x=self.capa2(x)
    return tf.nn.sigmoid(x)
  