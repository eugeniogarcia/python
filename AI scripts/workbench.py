from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential

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