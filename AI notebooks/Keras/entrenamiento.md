# Modelo basico

Podemos usar la API funcional para crear el modelo:

```py
def get_uncompiled_model():
  inputs = keras.Input(shape=(784,), name='digits')
  x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
  x = layers.Dense(64, activation='relu', name='dense_2')(x)
  outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
  model = keras.Model(inputs=inputs, outputs=outputs)
  return model
```

Para compilar el modelo tenemos que especificar cuatro cosas:

- __Algoritmo de optimizacion__ que vamos a utilizar
- __Funcion de error__ que sera el objetivo de la optimizacion
- __Metricas__ que deseamos calcular durante el proceso de optimizacion - opcional
- __Callbacks__ a ejecutar durante la optimizacion - opcional

Podemos especificar la funcion de error, metricas y callback bien usando el _tipo_ o el _nombre_. Por ejemplo, usando el tipo:

```py
def get_compiled_model_v1():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
  return model
```

Usando el nombre:

```py
def get_compiled_model_v2():
  model = get_uncompiled_model()
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
  return model
```

Con Keras tenemos varias funciones incluidas:

- Algoritmos de Optimizacion
	- SGD() (with or without momentum)
	- RMSprop()
	- Adam()
	- etc.
- Funciones de error
	- MeanSquaredError()
	- KLDivergence()
	- CosineSimilarity()
	- etc.
- Metricas
	- AUC()
	- Precision()
	- Recall()
	- etc.
- Callbacks:
	- ModelCheckpoint: Periodically save the model.
	- EarlyStopping: Stop training when training is no longer improving the validation metrics.
	- TensorBoard: periodically write model logs that can be visualized in TensorBoard (more details in the section "Visualization").
	- CSVLogger: streams loss and metrics data to a CSV file.
	- etc.

Keras nos permite crear funciones de error, metricas y callbacks __personalizados__. Tambien podemos crear capas y modelo personalizados. Empecemos viendo estos ultimos, para luego entrar a ver como personalizar las funciones de error, metricas y callbacks.

# Capa Personalizada (`layers.Layer`)

```py
from tensorflow.keras import layers
```

Si necesitamos definir una capa personalizada tendremos que extender la clase base que implementa una capa en Keras.

```py
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
```

En el constructor __init__ creamos la capa pasando todas sus propiedades. En el ejemplo anterior estamos indicando cual es la dimension de entrada y de salida. En el metodo __call__ se implementa la logica de procesamiento de la capa, como transforma la entrada en salida. En este ejemplo hemos creado tres variables __trainable__ que definen un modelo `input^2 * a + input * b + c`.

Hay una forma mas elegante de definir variables en una capa, utilizando el metodo __add_weight__. Este metodo crea una variable en la capa. Esta definicion es equivalente a la anterior:

```py
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
```

Podemos usar tambien variables que no se utilicen en el aprendizaje (variables independendientes):

```py
class ComputeSum(layers.Layer):

  def __init__(self, input_dim):
    super(ComputeSum, self).__init__()
    self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                             trainable=False)

  def call(self, inputs):
    self.total.assign_add(tf.reduce_sum(inputs, axis=0))
    return self.total
```

En este caso la capa se limita a acumular la suma de las entradas. Cada vez que se invoca la capa - se llamara a __call__ - se acumulara la suma de todos los elementos pasados como entrada.

Podemos ver que variables hay _trainables_ usando la propiedad __trainable_weights__ de la clase base:

```py
capa_cuadratica= Cuadratica_v2(4, 2)
y = capa_cuadratica(x)
print(capa_cuadratica.trainable_weights)

[<tf.Variable 'a:0' shape=(2, 4) dtype=float32, numpy=
array([[ 0.01215924,  0.046716  ,  0.029555  , -0.06673828],
       [ 0.03145774,  0.03732727, -0.06873614, -0.00882419]],
      dtype=float32)>, 
 <tf.Variable 'b:0' shape=(2, 4) dtype=float32, numpy=
array([[ 0.01828366,  0.01340114, -0.00576441, -0.01510664],
       [ 0.04222497,  0.04248193,  0.01320041,  0.08554146]],
      dtype=float32)>, 
 <tf.Variable 'c:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
```
Podemos ver todas las variables de la capa, entrenables o no, usando la propiedad __weights__ de la clase base:

```py
print(capa_cuadratica.weights)
```

## Best Practice. Diferir la creacion de las variables

En los ejemplos anteriores hemos incluido en el constructor el parametro que define la dimension de entrada. Esto hace que cuando se cree la capa se tenga que saber de antemano su dimension. En muchas ocasiones esto no es asi hasta el ultimo momento, cuando pasamos el training set a la capa. Para modelar nuestra capa de forma que no sea necesario especificar la dimension de entrada en el constructor, usaremos el metodo __build__ - este metodo tambien se usa en la personalizacion del modelo.

```py
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
```

Destacar lo siguiente:

- En el constructor no especificamos la dimension de entrada - antes _input_dim_
- En el constructor no se crean las variables. Guardamos en __self.units__ la dimension de salida. La utilizaremos mas tarde
- El metodo __build__ es el que declara las variables. 
	- El metodo tiene como argumento el imput de la capa
	- El metodo se llamara una sola vez, la primera vez que ejecutemos la capa - es decir, la primera vez que se llame a _call_
	- Las variables se declaran como antes. La dimension de entrada se obtiene como __input_shape[-1]__
- El metodo __call no sufre cambios__

Cuando creemos esta capa ya no especificamos la dimension de entrada:

```py
capa_cuadratica= Cuadratica_v3(4)
```

## Combinar capas

Las capas se pueden convinar. Por ejemplo, podriamos definir una capa linea, y luego convinarla para crear una cuadratica. La lineal seria:

```py
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
```

Nada nuevo en la definicion de esta capa. Veamos ahora como podemos convinar varias capas para dar lugar a otra. Aqui usamos dos lineales, con una activacion `relu` para dar lugar a una capa que deberia ser capaz de aprender un modelo cuadratico:

```py
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
```

Podemos usar la capa `Cuadratica_v4` como si tal cosa. Por ejemplo, podemos ver cuales son las variables entrenables:

```py
capa_cuadratica= Cuadratica_v4(4)
y = capa_cuadratica(x)
print(capa_cuadratica.trainable_weights)

[<tf.Variable 'cuadratica_v4/lineal_v1/a:0' shape=(2, 4) dtype=float32, numpy=
array([[-5.2751493e-02, -2.5294214e-03, -6.5789581e-03, -7.3815022e-06],
       [-2.1242544e-02, -6.9996528e-02,  3.0175557e-02,  4.1047480e-02]],
      dtype=float32)>, 
<tf.Variable 'cuadratica_v4/lineal_v1/b:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, 
<tf.Variable 'cuadratica_v4/lineal_v1_1/a:0' shape=(4, 4) dtype=float32, numpy=
array([[ 0.06250542, -0.02384896, -0.0481087 , -0.00968368],
       [-0.04035858, -0.00286195,  0.02488531,  0.00798782],
       [ 0.05453729, -0.02987812,  0.00787175, -0.03253619],
       [-0.01131698,  0.02975063, -0.0067396 , -0.01307838]],
      dtype=float32)>, 
<tf.Variable 'cuadratica_v4/lineal_v1_1/b:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
```

## Errores de la capa

Cuando entrenamos un modelo lo que hacemos en esencia es minimizar - optimizar - una funcion de error. La funcion de error que se especifica en el modelo - al compilarlo - utiliza la salida esperada del modelo frente a la salida esperada para calcular el error del modelo. 

En ciertas ocasiones nos puede interesar considerar en la funcion de error no solamente la salida del modelo sino algun estado interno asociado. Un ejemplo claro de esto es, sin ir mas lejos, lo que hacemos cuando añadimos el L1 o el L2 de los pesos como medida para evitar el overfitting. En estos casos durante el training lo que queremos favorecer es que los pesos sean lo más pequeños posibles para que las derivadas de las funciones de activacion no sea muy pequeñas - lo que hacer que a medida que se acumulan capas el efecto en el error de cada capa se vaya atenuando. En estos casos ademas de considerar el error como la diferencia entre la salida esperada y la real, añadimos otros factores a la funcion de error que tiene que ver con el valor absoluto de los pedos - L1 - o su cuadrado - L2. Para calcular cosas como estas debemos usar estados internos, en este caso las variables/pesos de las capas intermedias.

Veamos como al definir una capa podemos definir un error que se añadiria al error de salida del modelo, y por lo tanto considerado en la optimizacion que busca el minimo del error.

```py
class ErroresRegularizacion(layers.Layer):

  def __init__(self, tasa=1e-2):
    super(ErroresRegularizacion, self).__init__()
    self.tasa = tasa

  def call(self, inputs):
    self.add_loss(self.tasa * tf.reduce_sum(inputs))
    return inputs
```

En este caso la capa no _transforma_ la entrada, es decir, que en cierto modo es transparente en el calculo de la salida del modelo. Sin embargo con __add_loss__ estamos definiendo una contribucion de esta capa al error final del modelo. en este caso lo que esta capa haria es fomentar que los pesos sean lo mas pequeños posible.

Otra peculiaridad de los errores definidos en cada capa es que:

- Los errores se propagan a capas superiores. Esto es, si creamos una capa que combina otras capas, los errores de la capa _padre_ seran la adicion de los errores en la capas hijas
- Con cada calculo del modelo, con cada _forward-pass_ de la entrada por las capas del modelo, el error se resetea. Esto es, que con cada iteraccion de entrenamiento partimos con el contador de errores _a cero_.

Veamos una capa _padre_ de la que acabamos de definir:

```py
class CapaPadre(layers.Layer):

  def __init__(self):
    super(CapaPadre, self).__init__()
    self.activity_reg = ErroresRegularizacion(1e-2)

  def call(self, inputs):
    return self.activity_reg(inputs)
```

Antes de usar la capa la lista de errores estara vacia
```py
layer = CapaPadre()
assert len(layer.losses) == 0  # La primera vez los errores deben ser cero
```

Cuando la usamos se creara un error en la lista. Como la entrada es un escalar con valor 1, el error sera 0.01:

```py
_ = layer(tf.ones(1, 1))
assert len(layer.losses) == 1  
print(layer.losses[0].numpy()) # 1 * 1e-2

0.01
```

Si hacemos otra ejecucion la lista sigue teniendo un solo error - demostrando que entre ejecuciones los errores se resetean:

```py
_ = layer(tf.ones(1, 1))
assert len(layer.losses) == 1  
print(layer.losses[0].numpy()) # 1 * 1e-2

0.01
```

Destacar lo que ya se ha comentado, que en __layer.losses__ tenemos una lista con todos los errores acumulados por la capa y sus hijos (tantos elementos como errores se han acumulado; las suma de todos los elementos sera la contrinucion al error que hara esta capa.

### Error intermedio en training loops

Para reforzar lo que ya se ha indicado antes, veamos un training loop _custom_, para que asi podamos apreciar todas las partes involucradas en la definicion del error que tenemos que optimizar.

Supongamos que usamos __SparseCategoricalCrossentropy__ como funcion de error, y el __SGD__ como algoritmo de optimizacion:

```py
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

El error que tendremos que optimizar sera:
- El error de salida del modelo. Esto es, el error medido por SparseCategoricalCrossentropy
- Todos los errores _internos_ definidos por las capas del modelo

Vemoslo en accion. Para cada batch de datos vamos a calcular la derivada...

```py
for x_batch_train, y_batch_train in train_dataset:
  with tf.GradientTape() as tape:
```

...del error de salida del modelo. El error es la suma del error de salida del modelo...

```py
    logits = layer(x_batch_train)  # Logits for this minibatch
    loss_value = loss_fn(y_batch_train, logits)
```

...mas los errores de las capas internas:

```py
    loss_value += sum(model.losses)
```

Ya solo queda calcular la derivada del error con respecto a las variables _trainables_:

```py
  grads = tape.gradient(loss_value, model.trainable_weights)
```

Y proceder a optimizar las variables trainables en la direccion del gradiente:

```py  
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
```
  
# Modelo Personalizado (`tf.keras.Model`)

Habitualmente lo que haremos para crear nuestros modelos es usar

- capas incluidas en Keras
- definir capas personalizadas cuando no haya una capa que cubra nuestras necesidades
- usar `Sequential` o la `API funcional` para combinar las capas y crear el modelo que necesitemos entrenar
- usar la clase _Model_ cuando la logica de combinacion de capas no pueda ser satisfecha con c)

```py
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
```

# Personalizacion

## Personalizando la Funcion de error

Para personalizar la funcion de error tenemos tres metodos:

- Crear una funcion `python` que tome dos argumentos y retorne el error
- Crear una clase que extienda la funcion base de error
- Definiendo el error en la propia capa

### Usar una funcion `python`

Definimos una funcion:

```py
def basic_loss_function(y_true, y_pred):
    return tf.math.reduce_mean(y_true - y_pred)
```

Ahora usamos la funcion al compilar el modelo:

```py
model.compile(optimizer=keras.optimizers.Adam(),loss=basic_loss_function)
```

### Extendiendo la clase base (`keras.losses.Loss`)

Extendemos la clase base `keras.losses.Loss`. En el constructor pasamos todos los argumentos que necesitemos para inicializar la funcion:

```py
class miFuncionWeightedBinaryCrossEntropy(keras.losses.Loss):
    def __init__(self, pos_weight, weight, from_logits=False,
                 reduction=keras.losses.Reduction.AUTO,
                 name='weighted_binary_crossentropy'):
        super(miFuncionWeightedBinaryCrossEntropy, self).__init__(reduction=reduction, name=name)
        self.pos_weight = pos_weight
        self.weight = weight
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if not self.from_logits:
            x_1 = y_true * self.pos_weight * -tf.math.log(y_pred + 1e-6)
            x_2 = (1 - y_true) * -tf.math.log(1 - y_pred + 1e-6)
            return tf.add(x_1, x_2) * self.weight 
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.pos_weight) * self.weight
```

La funcion __call__ implementa la funcion de errores. En este caso estamos definiendo una funcion de clasificacion binaria. Una vez definida la clase la podemos utilizar:

```py
model.compile(optimizer=keras.optimizers.Adam(),loss=miFuncionWeightedBinaryCrossEntropy(0.5, 2))
```

### Definiendo el error en la propia capa

Si el error que queremos definir no depende de la salida del modelo, si depende de otros cosas, por ejemplo del estado de una determinada capa, los dos metodos anteriores no nos van a ayudar. En este caso podemos aprovecharnos - de lo que ya comentamos al ver la definicion de capas personalizadas -, del error de la capa - o del modelo.

```py
class ActivityRegularizationLayer(layers.Layer):

  def call(self, inputs):
    self.add_loss(tf.reduce_sum(inputs) * 0.1)
    return inputs 
```

Aqui podemos ver como hemos usado la funcion __add_loss__ para añadir un error. Este error se unira al resto de errores definidos en las capas, y al error del modelo.

## Personalizando las metricas

Para crear una metrica personalizada tenemos dos metodos:

- Crear una clase que extienda la funcion base de metricas
- Definiendo la metrica en la propia capa o a nivel de modelo

### Extendiendo la clase base (`keras.metrics.Metric`)

Extendemos la clase base `keras.metrics.Metric`:
```py
class miMetrica(keras.metrics.Metric):

    def __init__(self, name='categorical_true_positives', **kwargs):
      super(miMetrica, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
      values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
      values = tf.cast(values, 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.true_positives

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.true_positives.assign(0.)
```

Para crear una metrica de este modo hay que seguir los siguientes pasos:

- Usar el constructor para pasar aquellos argumentos que necesite nuestra metrica, y definir la variable en la que guardaremos la metrica
- Implementar los metodos:
	- reset_states. Inicializar la metrica. Se invocara con cada epoch de training
	- update_state. Actualizar la metrica. Se invocara durante el entrenamiento del modelo
	- result. Recupera el valor de la metrica
	
En el constructor creamos la variable en la que se guardara la metrica usando el metodo __add_weight__:

```py
    def __init__(self, name='mi_categorical_true_positives', **kwargs):
      super(miMetrica, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')
```

Con __reset_states__ inicializamos la metrica. En este caso le damos el valor 0:

```py
self.true_positives.assign(0.)
```

Durante el entrenamiento del modelo se actualizara la metrica usando el metodo __update_state__:

```py
    def update_state(self, y_true, y_pred, sample_weight=None):
	  # Crea un tensor (None,1) con el valor maximo de cada muestra
      y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
      values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
      values = tf.cast(values, 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))
```

Finalmente debemos implementar __result__ para obtener el valor de la metrica:

```py
return self.true_positives
```

Para utilizar la metrica custom:

```py
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[miMetrica()])
```

Durante el entrenamiento se calculara esta metrica:

```py
Train on 50000 samples
Epoch 1/3
50000/50000 [==============================] - 3s 50us/sample - loss: 0.2016 - mi_categorical_true_positives: 46924.0000
Epoch 2/3
50000/50000 [==============================] - 2s 42us/sample - loss: 0.0778 - mi_categorical_true_positives: 48838.0000
Epoch 3/3
50000/50000 [==============================] - 2s 41us/sample - loss: 0.0656 - mi_categorical_true_positives: 49010.0000
```

### Definiendo el error en la propia capa o en el modelo

```py
class MetricLoggingLayer(layers.Layer):

  def call(self, inputs):
    self.add_metric(keras.backend.std(inputs),
                    name='mi_std_of_activation',
                    aggregation='mean')
    return inputs 
```

Al utilizar esta capa se incluira la metrica _mi_std_of_activation_ - y nada mas, porque la capa hace solo de pass-through. La metrica calcula la media de los _keras.backend.std(inputs)_.

```py
Train on 50000 samples
50000/50000 [==============================] - 2s 50us/sample - loss: 0.3394 - mi_std_of_activation: 0.9428
```

Podriamos usar el modelo en lugar de la capa:

```py
inputs = keras.Input(shape=(784,), name='digits')
x1 = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x2 = layers.Dense(64, activation='relu', name='dense_2')(x1)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)

model.add_metric(keras.backend.std(x1),
                 name='std_of_activation',
                 aggregation='mean')

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
              loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train,
          batch_size=64,
          epochs=1)
```

Observese como usamos __model.add_loss__ y __model.add_metric__.

# Validacion

Durante el entrenamiento del modelo debemos apartar una serie de datos de entrada para la validacion - y que no se usen para el entrenamiento. Podemos hacer esto de tres formas:

- Cuando se usen _numpy arrays_ podemos simplemente especificarlo con un % - en este caso 0.2 o lo que es lo mismo, 20%:

```py
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=3)
```

- Usando dos sets de _numpy arrays_. Un par para entrenamiento y otro para validacion:
- Usando dos datasets, uno para entrenamiento y otro para validacion:

```py
model = get_compiled_model()

# Prepare the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(64)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
```

```py
model.fit(train_dataset, epochs=3,validation_data=val_dataset, validation_steps=10)
```

# Sample weighting and class weighting

