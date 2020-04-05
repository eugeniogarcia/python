# Introducción

Con tf.estimator tenemos acceso a una serie de modelos que podemos reutilizar. Para utilizar un modelo de tf.estimator hay un procedimiento que tenemos que seguír para preparar los datos, entrenar el modelo y luego evaluarlo.

## Escribir una funcion que importe un dataset

Tenemos que crear una función que devuelva un dataset que sea una dupla (diccionario de features,etiqueta). El patron es el siguiente:

```py
def input_fn(dataset):
    ...  # Manipular el dataset - desordenar, definir el batch, etc. y devolver un diccionario y una etiqueta
    return feature_dict, label
```

Con esta función estamos adecuando un Pandas Dataframe en un dataset. No será la única cosa que tengamos que hacer...

## Definir las columnas de features

En el dataset tendremos un diccionario con features y una etiqueta que utilizaremos para adiestrar nuestro modelo. Antes de poder hacerlo tendremos que adaptar las features. Para hacerlo usaremos `tf.feature_column`. Estas funciones nos permitirán transformar una feature antes de utilizarla en el modelo. Existen diferentes prototipos que podremos utilizar para gestionar distintos tipos de features. En todos los casos el prototipo utilizara el nombre de la feature - el nombre de la key del diccionario del dataset que vamos a alimentar en el modelo.

En el ejemplo siguiente definimos tres features numéricas llamadas `population`, `crime_rate` y `median_education`. La tercera feature es preprocesada con un lambda.

```py
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',normalizer_fn=lambda x: x - global_education_mean)
```

### Feature Categorica

Cuando tengamos una feature que haya que codificar one shot, utilizaremos esta `feature_column`:

```py
tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary)
```

Especificamos por un lado el nombre de la feature, `feature_name`, que será el nombre de la serie en los datos que queremos pre-procesar. En `vocabulary` especificaremos el dominio de valores a categorizar.

#### Forma alternativa de hacer el one-shot

Con Pandas tenemos la opción de poder convertir una columna categorica a one-shot. Lo podemos ver en acción con la Serie sex. Más adelante vemos el uso más estandar con feature_columns:

```py
dftrain['sex'] = pd.Categorical(dftrain['sex'])
dftrain['sex'] = dftrain.sex.cat.codes
dfeval['sex'] = pd.Categorical(dfeval['sex'])
dfeval['sex'] = dfeval.sex.cat.codes
```

Esto crear el one-shot encoding con la serie `sex`.

### Numericas

Para las features que sean de tipo numérico el modelado se hará de esta forma:

```py
tf.feature_column.numeric_column(feature_name, dtype=tf.float32)
```

Podremos aplicar una lambda para preparar los datos:

```py
tf.feature_column.numeric_column(feature_name, dtype=tf.float32, normalizer_fn=lambda x: x - global_education_mean)
```

Especificamos por un lado el nombre de la feature, `feature_name`, que será el nombre de la serie en los datos que queremos pre-procesar. Por otro lado indicamos el tipo de dato.

### Buckets

Cuando necesitamos transformar valores en un rango de valores, buckets, esta será la `feature_column` a utilizar:

```py
tf.feature_column.numeric_column(feature_name, boundaries=[])
```

Por ejemplo, en este caso estamos agrupando la feature `age` en diez buckets (0 a 18, 19 a 25, 26 a 30, ...):

```py
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

### Embeding

En caso de que el número de categorias sea elabado no sería aconsejable una categorización one shot. El embeding permite categorizar los datos pero sin necesitar tantas categorias. En este ejemplo estamos clasificando los datos en 8 categorias - eso si, no son booleanas:

```py
thal_embedding = feature_column.embedding_column(thal, dimension=8)
```

### Hashing

Otra opción para codificar una feature, limitando el número de categorias, es usar el valor hasheado. Por ejemplo en este caso el hash es de tamaño 1000:

```py
thal_hashed = feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
```

### Derivada

Cuando necesitamos combinar dos features en una sola podemos utilizar la `feature_column` `crossed_column`:

```py
age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
```

### Utilizar features

Para utilizar estas `feature_column` lo haremos como funciones, a las que pasaremos el dataset:

```py
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch)
```

Podemos convertir el resultado a un `numpy`:

```py
tf.keras.layers.DenseFeatures([tf.feature_column.indicator_column(gender_column)])(feature_batch).numpy()
```

Aquí podemos ver como pasar las features al modelo:

```py
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
```

## Utilizar un modelo

Finalmente tendremos que seleccionar un modelo. Por ejemplo, un `LinearClassifier`:

```py
estimator = tf.estimator.LinearClassifier(feature_columns=[population, crime_rate, median_education])
```