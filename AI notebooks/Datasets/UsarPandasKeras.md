# Introduccion

Podemos usar Pandas dataframes para adiestrar modelos Keras. Hay que tener una serie de consideraciones

## tf.estimator

Si usamos Dataframes con modelos de tf.estimator lo que haremos será el dataset de la siguiente forma:

```py
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
```

Creamos un diccionario con el dataframe, `dict(data_df)`. El diccionario lo usamos para crear una dupla, `(dict(data_df), label_df)`, que luego convertimos en dataset con `from_tensor_slices()`.

Este dataset lo podemo usar para alimentar el modelo de tf.estimator, pero nos daría problemas para usarlo con un modelo Keras directamente. Esto es porque salvo que espeficificamente indiquemos cuales son las keys del diccionario de entrada al modelo Keras, Keras espera unos nombres para las claves del diccionario que no coinciden con los nombres del diccionario que hemos creado aqui - y que utiliza como keys los nombres de las columnas/series de nuestro Dataframe Pandas.

Para evitar este problema hay dos formas que veremos en la siguiente sección.

## Keras

### Usar `tf.data.Dataset.from_tensor_slices` para leer los valores del Panda dataframe

```py
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
```

### Utilizar un diccionario como entrada al modelo

Creamos un diccionario que usaremos como entrada. Recorremos todas las keys del Dataframe `for key in df.keys()`, y creamos un diccionario que tiene como key el nombre de la serie en pandas y como valor una `tf.keras.layers.Input` con el nombre de la serie en pandas. Finalmente creamos una lista con este objeto `list(inputs.values())`.

```py
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}

x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=['accuracy'])
```