import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=4)

############################################
#Definir datasets en memoria
############################################
#Un dataset con escalares, tres
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0])

print("Recupera los datos:")
for a in dataset:
  print(a.numpy())
  
#Un dataset con tres elementos de dos dimensiones
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([3, 2], minval=1, maxval=10, dtype=tf.int32))

#Son 3 datos con 2 features cada uno:
print("Recupera los datos con un iterator")
it=iter(dataset1)
for a in it:
  print(a.numpy())


dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([3]),
                                               tf.random.uniform([3, 2], maxval=100, dtype=tf.int32)
                                               ))

#Son 3 datos, cada uno un diccionario de dos elementos. EL primer elemento es un escalar, y el segundo un vector de dimention 2
print("Recupera los datos con un iterator")
for a,b in dataset2:
#  print("{a.numpy()} - {b.numpy()}".format(a=a,b=b))
  print("{a} - {b}".format(a=a,b=b))

print("Recupera los datos con un iterator")
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
for a,(b,c) in dataset3:
#  print("{a.numpy()} - {b.numpy()}".format(a=a,b=b))
  print("{a} x {b} x {c}".format(a=a,b=b,c=c))

#Sparse tensor
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

############################################
#Definir un dataset a partir de un generador
############################################
def count(stop):
  i = 0
  while i<stop:
    yield i
    i += 1
    
ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

#Repeat. Cuando se hayan extraido todos los datos del dataset, se resetea el dataset y se puede volver a usar desde el principio
#Suffle. Ordena de forma aleatoria
#Batch. Cada extraccion de datos recupera un bloque de datos. En este ejemplo sacamos 7
#Take. Determina cuantos bloques de datos sacar. En este caso 5
print("Recupera los datos desde un generador")
for count_batch in ds_counter.repeat().shuffle(buffer_size=2).batch(7).take(3):
  print(count_batch.numpy())


#Vamos a generar un dataset con datos heterogeneos en su shape
val=[[8, 3, 0],[8, 3, 0,2],[8, 3],[8],[1, 5, 0],[6, 8, 0,3],[5, 3],[9]]

def genera(stop):
  i = 0
  while i<stop:
    #yield val[np.random.randint(0,7)]
    yield val[i]
    i += 1
    if(i>7):
      i=0

dataset5 = tf.data.Dataset.from_generator(genera, args=[25], output_types=tf.int32)

#No podemos usar batch porque cada dato tiene un shape diferente
print("Recupera los datos desde un generador")
for count_batch in dataset5.repeat().take(3):
  print(count_batch.numpy())

#Parecido, pero en este caso generamos los batch con padding, porque cada muestra tiene un shape diferente
for count_batch in dataset5.repeat().padded_batch(7, padded_shapes=([4])).take(3):
  print(count_batch.numpy())


############################################
#Definir un dataset a partir de archivos
############################################
flores = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)


generador_imagenes = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

images, labels = next(generador_imagenes.flow_from_directory(flores))