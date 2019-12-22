from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import VGG16

#########################################
#Images Processing
#########################################

model_images = Sequential()

model_images.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_images.add(layers.MaxPooling2D((2, 2)))
model_images.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_images.add(layers.MaxPooling2D((2, 2)))
model_images.add(layers.Conv2D(64, (3, 3), activation='relu'))

model_images.add(layers.Flatten())
model_images.add(layers.Dense(64, activation='relu'))
model_images.add(layers.Dropout(0.5))
model_images.add(layers.Dense(10, activation='softmax'))

print("Modelo para procesar imagenes")
print(model_images.summary())


#########################################
#Reusar un modelo existente
#########################################

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)) 

model_reusar = Sequential()

model_reusar.add(conv_base)
model_reusar.add(layers.Flatten())
model_reusar.add(layers.Dense(256, activation='relu'))
model_reusar.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print("Modelo para procesar imagenes reusando el VGG16")        
print(model_reusar.summary())


#########################################
#Text Processing
#One-hot encode
#########################################

#Tamaño del diccionario
max_features=1000
#Tamaño de las frases
maxlen=20


model_hot_encode = Sequential()

#La entrada es un vector de max_features, con 1 en aqullas palabras que se encuentren entre las maxlen palabras de la frase (None, max_features)
model_hot_encode.add(layers.Dense(8, activation='relu', input_shape=(max_features,)))
model_hot_encode.add(layers.Dense(32, activation='relu'))
model_hot_encode.add(layers.Dense(1, activation='sigmoid'))

print("Modelo para procesar texto con hot encoding")  
print(model_hot_encode.summary())


#########################################
#Text Processing
#Embeding
#########################################

model_embeded = Sequential()

#La entrada es un vector de tamaño maxlen, con cada valor un entero que representa la palabra (None, maxlen)
model_embeded.add(layers.Embedding(max_features, 8, input_length=maxlen,))
model_embeded.add(layers.Flatten())
model_embeded.add(layers.Dense(32, activation='relu'))
model_embeded.add(layers.Dense(1, activation='sigmoid'))

print("Modelo para procesar texto con embedding")  
print(model_embeded.summary())


#########################################
#Text Processing
#Embeding and Recurrent Networks
#########################################

model_recurrente = Sequential()

#La entrada es un vector de tamaño maxlen, con cada valor un entero que representa la palabra (None, maxlen)
model_recurrente.add(layers.Embedding(max_features, 8, input_length=maxlen))
model_recurrente.add(layers.LSTM(32))
model_recurrente.add(layers.Dense(1, activation='sigmoid'))

print("Modelo para procesar texto con red recurrente")  
print(model_recurrente.summary())


#########################################
#Text Processing
#Embeding and Recurrent Networks (con dropout)
#########################################

model_recurrente = Sequential()

#La entrada es un vector de tamaño maxlen, con cada valor un entero que representa la palabra (None, maxlen)
model_recurrente.add(layers.Embedding(max_features, 8, input_length=maxlen))
model_recurrente.add(layers.LSTM(32,dropout=0.2,recurrent_dropout=0.2))
model_recurrente.add(layers.Dense(1, activation='sigmoid'))

print("Modelo para procesar texto con red recurrente")  
print(model_recurrente.summary())


#########################################
#Text Processing
#Embeding and Recurrent Networks (varias capas)
#########################################

model_recurrente = Sequential()

#La entrada es un vector de tamaño maxlen, con cada valor un entero que representa la palabra (None, maxlen)
model_recurrente.add(layers.Embedding(max_features, 8, input_length=maxlen))
model_recurrente.add(layers.LSTM(32,return_sequences=True))
model_recurrente.add(layers.LSTM(32,return_sequences=True))
model_recurrente.add(layers.LSTM(32))
model_recurrente.add(layers.Dense(1, activation='sigmoid'))

print("Modelo para procesar texto con red recurrente (varias capas")  
print(model_recurrente.summary())


#########################################
#Text Processing
#Convolution (1D)
#########################################

model_conv_1d = Sequential()

#La entrada es un vector de tamaño maxlen, con cada valor un entero que representa la palabra (None, maxlen)
model_conv_1d.add(layers.Embedding(max_features, 8, input_length=maxlen))
model_conv_1d.add(layers.Conv1D(32, 3, activation='relu'))
model_conv_1d.add(layers.MaxPooling1D(2))
model_conv_1d.add(layers.Conv1D(32, 3, activation='relu'))
model_conv_1d.add(layers.GlobalMaxPooling1D())
model_conv_1d.add(layers.Dense(1))

print("Modelo para procesar texto con convolucion 1D")  
print(model_conv_1d.summary())
