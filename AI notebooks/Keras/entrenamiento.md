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

Las capas se pueden combinar. Por ejemplo, podriamos definir una capa linea, y luego combinarla para crear una cuadratica. La lineal seria:

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

Nada nuevo en la definicion de esta capa. Veamos ahora como podemos combinar varias capas para dar lugar a otra. Aqui usamos dos lineales, con una activacion `relu` para dar lugar a una capa que deberia ser capaz de aprender un modelo cuadratico:

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

En ciertas ocasiones nos puede interesar considerar en la funcion de error no solamente la salida del modelo sino algun estado interno asociado. Un ejemplo claro de esto es, sin ir mas lejos, lo que hacemos cuando a�adimos el L1 o el L2 de los pesos como medida para evitar el overfitting. En estos casos durante el training lo que queremos favorecer es que los pesos sean lo m�s peque�os posibles para que las derivadas de las funciones de activacion no sea muy peque�as - lo que hacer que a medida que se acumulan capas el efecto en el error de cada capa se vaya atenuando. En estos casos ademas de considerar el error como la diferencia entre la salida esperada y la real, a�adimos otros factores a la funcion de error que tiene que ver con el valor absoluto de los pedos - L1 - o su cuadrado - L2. Para calcular cosas como estas debemos usar estados internos, en este caso las variables/pesos de las capas intermedias.

Veamos como al definir una capa podemos definir un error que se a�adiria al error de salida del modelo, y por lo tanto considerado en la optimizacion que busca el minimo del error.

```py
class ErroresRegularizacion(layers.Layer):

  def __init__(self, tasa=1e-2):
    super(ErroresRegularizacion, self).__init__()
    self.tasa = tasa

  def call(self, inputs):
    self.add_loss(self.tasa * tf.reduce_sum(inputs))
    return inputs
```

En este caso la capa no _transforma_ la entrada, es decir, que en cierto modo es transparente en el calculo de la salida del modelo. Sin embargo con __add_loss__ estamos definiendo una contribucion de esta capa al error final del modelo. en este caso lo que esta capa haria es fomentar que los pesos sean lo mas peque�os posible.

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

# Otras Personalizaciones

Ademas de las capas y del modelo podemos personalizar:

- Funciones de error
- Metricas
- Callbacks

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

Aqui podemos ver como hemos usado la funcion __add_loss__ para a�adir un error. Este error se unira al resto de errores definidos en las capas, y al error del modelo.

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

## Personalizando Callbacks (`tf.keras.callbacks.Callback`)

Extendemos la clase `tf.keras.callbacks.Callback`:

```py
class miCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
```

Ya podemos usar el callback:

```py
model = get_model()
_ = model.fit(x_train, y_train,batch_size=64,epochs=1,steps_per_epoch=5,verbose=0,
          callbacks=[miCustomCallback()])
```

Podemos usar callbacks en los siguientes metodos:

- fit(), fit_generator()
- evaluate(), evaluate_generator()
- predict(), predict_generator()

```py
_ = model.evaluate(x_test, y_test, batch_size=128, verbose=0, steps=5,
          callbacks=[miCustomCallback()])
```

- on_(train|test|predict)_begin(self, logs=None)
Called at the beginning of fit/evaluate/predict.

- on_(train|test|predict)_end(self, logs=None)
Called at the end of fit/evaluate/predict.

- on_(train|test|predict)_batch_begin(self, batch, logs=None)
Called right before processing a batch during training/testing/predicting. Within this method, logs is a dict with batch and size available keys, representing the current batch number and the size of the batch.

- on_(train|test|predict)_batch_end(self, batch, logs=None)
Called at the end of training/testing/predicting a batch. Within this method, logs is a dict containing the stateful metrics result.

In addition, for training, following are provided.

- on_epoch_begin(self, epoch, logs=None)
Called at the beginning of an epoch during training.

- on_epoch_end(self, epoch, logs=None)
Called at the end of an epoch during training.

### Parar el entrenamiento

```py
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

  def __init__(self, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()

    self.patience = patience

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('loss')
    if np.less(current, self.best):
      self.best = current
      self.wait = 0
      # Record the best weights if current results is better (less).
      self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        #Paramos el entrenamiento
        self.model.stop_training = True
        print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
```

Notese como con __self.model.stop_training__ podemos para el entrenamiento.

### Caso practico

Vamos a mostrar como utilizar dos callbacks, el de tensorflow y el de early stop. El de tensorflow:

```py
tensorboard_cbk=keras.callbacks.TensorBoard(
  log_dir='.\\entrena',
  histogram_freq=1, 
  update_freq='epoch', # indica que se generaran muestras de datos por cada epoch
#  update_freq='batch', # indica que se generaran muestras de datos por cada batch
  profile_batch = 0)
```

Lo mas destacado aqui es que tuve que incluir `profile_batch = 0` porque de lo contrario se disparaba un error que indicaba que no se pudo parar el profiler. El callback de parada temprana:

```py
#Callbacks. Parada anticipada del entrenamiento
paradaTemprana_cbk=keras.callbacks.EarlyStopping(
        # Observa la funcion de perdida - tipicamente usariamos val_loss en lugar de loss
        monitor='val_loss',
        # Si la funcion de error cambia menos de min_delta
        min_delta=1e-3,
        # en patience batchs, para
        patience=6,
        verbose=1)
```

Finalmente los utilizamos:

```py
modelo.fit(train_dataset,validation_data=train_dataset_val, epochs=ciclos_entrenamiento,callbacks=[paradaTemprana_cbk,tensorboard_cbk])
```

Para ver los resultados en tensorflow:

```sh
tensorboard --logdir=./entrena
```

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

Para aquellos casos en los que necesitemos dar mas importancia a unos datos de prueba sobre otros, la solucion es usar `weighting`. Podamos usar `weighting` de dos formas:

- En los datos de prueba. Determinados datos de entrada toman mas importancia en el calculo de la salida
- En la clases de salida. Un error en las clases de salida tiene diferente peso para unas clases que otras

Si entrenamos el modelo con datos de __numpy__ usaremos los argumentos __sample_weight__ y __class_weight__. Si entrenamos con __Datasets__ usaremos una tupla __(input_batch, target_batch, sample_weight_batch)__.

En este ejemplo damos preeminencia a la clase _5_. En este ejemplo el peso de la clase 5 es el doble que el resto:

```py
class_weight = {0: 1., 1: 1., 2: 1., 3: 1., 4: 1.,
                5: 2.,
                6: 1., 7: 1., 8: 1., 9: 1.}

model.fit(x_train, y_train,
          class_weight=class_weight,
          batch_size=64,
          epochs=4)
```

Podriamos usar esta tecnica si, por ejemplo, tuvieramos menos datos para entrenar para la clase 5. En el siguiente ejemplo usamos pesos para los datos de entrada para hacer algo equivalente:

```py

# Identificamos aquellos casos de prueba que corresponden a la clase `5`, y les asignamos un peso doble
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.

model = get_compiled_model()
model.fit(x_train, y_train,
          sample_weight=sample_weight,
          batch_size=64,
          epochs=4)
```

Los dos ejemplo anteriores son con datos de �numpy`. Con `Datasets`:

```py
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.

train_dataset = tf.data.Dataset.from_tensor_slices( (x_train, y_train, sample_weight) )

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

model = get_compiled_model()
model.fit(train_dataset, epochs=3)
```

# Modelos con multiples entradas y salidas

## Definir el modelo

Podemos usar modelos con varias entradas y/o salidas usando la api funcional de Keras. En este ejemplo vamos a definir dos entradas:

- Una imagen 32x32 con 3 canales
- Un time series de vectores con 10 features, y con `None` series. No hemos especificado cuantas instancias tiene la serie, de ahi el `None`.

```py
from tensorflow import keras
from tensorflow.keras import layers

image_input = keras.Input(shape=(32, 32, 3), name='img_input')
timeseries_input = keras.Input(shape=(None, 10), name='ts_input')
```

Usamos a continuacion ambas entradas para alimentar el modelo:

```py
x1 = layers.Conv2D(3, 3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3, 3)(timeseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)
```

Y las salidas de ambos convergen ahora. En este caso las concatenamos:

```py
x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, activation='softmax', name='class_output')(x)
```

Notese como hemos especificado el nombre a las dos capas de salida. El modelo se crea:

```py
model = keras.Model(inputs=[image_input, timeseries_input],
                    outputs=[score_output, class_output])
```

Podemos visualizar el modelo:

```py
keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
```

## Compilar el modelo

A la hora de compilar el modelo

```py
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(),
          keras.losses.CategoricalCrossentropy()],
    metrics=[[keras.metrics.MeanAbsolutePercentageError(),
              keras.metrics.MeanAbsoluteError()],
             [keras.metrics.CategoricalAccuracy()]])
```

o de forma equivalente:

```py
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy()},
    metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                              keras.metrics.MeanAbsoluteError()],
             'class_output': [keras.metrics.CategoricalAccuracy()]})
```

Podemos tambien especificar que la contribucion al error de cada _rama_ del modelo no sea igual. En este ejemplo hacemos que `score_output` pesa el doble que `class_output`.

```py
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'score_output': keras.losses.MeanSquaredError(),
          'class_output': keras.losses.CategoricalCrossentropy()},
    metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                              keras.metrics.MeanAbsoluteError()],
             'class_output': [keras.metrics.CategoricalAccuracy()]},
    loss_weights={'score_output': 2., 'class_output': 1.})
```

# Tasa de aprendizaje variable

Una tecnica de optimizacion muy efectiva es definir una tasa de aprendizaje variable. Un caso habitual es hacer que la tasa de aprendizaje vaya disminuyendo, lo que se denomina __learning rate decay__. La tasa en que se reduce la _learning rate_ puede ser constante o variable. Por ejemplo vamos a reducirla en tasas de 96% en 10000 pasos:

```py
initial_learning_rate = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
```

Ahora especificamos esta definicion en lugar de pasar un valor fijo:

```py
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.mean_squared_error,tf.keras.metrics.Accuracy()])
```

Otras opciones disponibles son:

- ExponentialDecay
- PiecewiseConstantDecay
- PolynomialDecay
- InverseTimeDecay

Si quisieramos basar la tasa de aprendizaje en otro factor, por ejemplo, el _validation loss_, tendriamos que usar un callback para ajustarla, ya que el optimizador no tiene acceso al _validation loss_. Esto es lo que hace un callback como __ReduceLROnPlateau callback__.