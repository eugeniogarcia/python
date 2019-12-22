from tensorflow.keras import layers, Sequential

#Images Processing
model_images = Sequential()

model_images.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_images.add(layers.MaxPooling2D((2, 2)))
model_images.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_images.add(layers.MaxPooling2D((2, 2)))
model_images.add(layers.Conv2D(64, (3, 3), activation='relu'))

model_images.add(layers.Flatten())
model_images.add(layers.Dense(64, activation='relu'))
model_images.add(layers.Dense(10, activation='softmax'))

print(model_images.summary())





#Ejemplo de modelos
max_features=1000

model = Sequential()

model.add(layers.Embedding(max_features, 128, input_length=200))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.compile
print(model.summary())



#Dataset