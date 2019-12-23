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

## Funcion de error

Hay tres formas en las que podemos personalizar la funcion de error:

- Crear una funcion de python
- Extender la clase base que implementa la funcion de error
- Incluir como parte de la definicion de una capa personalizada la definicion del error
- Incluir como parte de la definicion del modelo personalizado la definicion del error