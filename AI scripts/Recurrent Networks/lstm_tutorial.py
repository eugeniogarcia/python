import tensorflow as tf
import numpy as np
import collections
import os
import argparse
import datetime as dt

"""To run this code, you'll need to first download and extract the text dataset
    from here: http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz. Change the
    data_path variable below to your local exraction path"""

data_path = "C:\\Users\\Eugenio\\Downloads\\datasamples\\data"

parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer: 1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The full path of the training data')
args = parser.parse_args()

#Lee el contenido de un archivo
def read_words(filename):
    #Lee el contenido de un archivo
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()

#Devuelve un diccionario que contiene cada palabra del archivo proporcionado como entrada,
#y le asigna como valor un id a cada palabra
def build_vocab(filename):
    #Lee el archivo
    data = read_words(filename)

    #Crea un diccionario estableciendo como key cada palabra, y con el valor igual 
    #al contador con la frecuencia de cada palabra
    #data podria ser una lista, un diccionario, 
    #>>> c = Counter()                           # a new, empty counter
    #>>> c = Counter('gallahad')                 # a new counter from an iterable
    #>>> c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
    #>>> c = Counter(cats=4, dogs=8)             # a new counter from keyword args
    #Podemos actualizar la coleccion con .update(data)
    #Podemos obtener una lista con los palabras en la coleccion con .elements(), repeating each as many times as its count
    #.most_common([n]) devuelve una lista con las duplas más habituales (palabra, frecuencia)
    #Podemos hacer aritmetica, +, -, 
    #.items() retorna una lista con el contenido del diccionario, [(palabra, frecuencia), (palabra, frecuencia),...]
    #.items(), .keys(), .values() son metodos habituales en el diccionario
    
    #Crea un diccionario con cada palabra como key, y la frecuencia de cada palabra en el archivo
    counter = collections.Counter(data)
    
    #sorted nos permite ordenar una lista, iterator, diccionario
    # sorted(iterable[, key][, reverse])
    # reverse=true hace que cambiemos el orden de la lista
    # Con key podemos pasar una funcion que define el criterio de ordenacion
    
    #Ordena el diccionario de mas a menos frecuencia
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    #Con zip(*diccionario) deshacemos el zip, retornara las partes constituyentes del zip
    #Lo que estamos haciendo aqui es crear una lista con todas las palabras, ordenadas de mas a menos frecuencia
    
    #Obtiene una lista con todas las palabras ordenadas por frecuencia, de mayor a menor
    words, _ = list(zip(*count_pairs))
    
    #zip crea un diccionario con tuplas formadas con cada uno de los elementos que hemos pasado como argumento
    #En este caso hemos creado un diccionario (dict), con duplas que tienen un elemento de words, y un numero
    #Si hay tres palabras en words, a, b y c, {(a,0),(b,1),(c,2)}
    
    #Construye un diccionario, con las palabras ordenadas por frecuencia, y como valor un id
    word_to_id = dict(zip(words, range(len(words))))

    #devuelve el diccionario
    return word_to_id


#Toma un archivo y devuelve una lista con los ids de cada palabra contenido en el archivo
#Si la palabra no estuviera se la salta
def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

#Carga los datos de adiestramiento, validacion, y prueba. Los devuelve como listas de ids - enteros
#Retorna tambien el diccionario y el diccionario inverso - relacion palabra con id
def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    #Lo que hacemos es unir las diez primeras palabras de train_data separadas por espacios
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    
    #Devuelve una lista con los ids que representan cada palabra del juego
    #de palabras para adiestrar, validar y probar, 
    #Devuelve tambien el diccionario, y el diccionario inverso
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


#Steps es el orden temporal. Step 2 significaria que en cada vector tenemos que registrar el valor en t, y en t-1
#batch_size es el numero de vectores que cosideramos por cada interaccion

def batch_producer(raw_data, batch_size, num_steps):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    #Tamaño total de los datos
    data_len = tf.size(raw_data)

    #Numero de interacciones para tratar todos los datos
    batch_len = data_len // batch_size
    
    #Oraganiza los datos en un tensor. La primera dimension es el numero de datos por extraccion, la segunda el numero de extracciones
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    #En una iteraccion, cuantos vectores puedo datos puedo confeccionar de Steps
    #Por ejemplo, si el batch es de tamaño 4, y el step es de 2, tendriamos que en el batch b1, b2, b3, b4. Los datos que podria preparar seria
    #3/2= 1
    #x valdria
    #b1, b2
    #b2, b3
    #y valdria
    #b2, b3
    #b3, b4
    
    epoch_size = (batch_len - 1) // num_steps

    #Siguiendo con el ejemplo anterior, i seria 0, 1
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    
    #data[:,0:2]
    #data[:,2:4]
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    
    #data[:,1:3]
    #data[:,3:5]
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])
    
    return x, y


class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = batch_producer(data, batch_size, num_steps)


# create the main model
class Model(object):
    def __init__(self, input, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input
        self.batch_size = input.batch_size
        self.num_steps = input.num_steps
        self.hidden_size = hidden_size

        # create the word embeddings
        with tf.device("/cpu:0"):
            #El peso W de la capa de embeding es un tensor de dos dimensiones: el tamaño del vocabulario, y el las dimensiones del 
            #espacio embebido.
            #input * W retornara la codificacion de cada vector de entrada hot-encoded segun el diccionario usando el vovabulario
            #se traducira en un vector con las dimensiones del espacio embebido
            #El espacio embebido tiene dimension self.hidden_size
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        #Si estamos entrenando la red, añadimos un dropout
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # set up the state storage / extraction
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        #toma el tensor self.init_state, y con cada iteraccion recupera un tensor de tamaño [2, self.batch_size, self.hidden_size]
        #Basicamente toma el axis indicado como pivote. En cada iteraccion sacaria [:,:...i..., :,:], donde i es el axis especificado
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        #Contiene la pareja de estados por cada capa. La pareja de estados es state + output de la capa anterior
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # create an LSTM cell to be unrolled
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
        
        # add a dropout wrapper if training
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        
        #tf.nn.dynamic_rnn construye la red neuronal como tal. Tiene como entrada las celdas LTSM, - es decir la estructura que modela la memoria -,
        #y el propio input. El ultimo argumento es la estructura en la que guardaremos los estados en cada capa
        #Las salidas son la salida de la red, que sera de tamaño [batch_size,num_steps,hidden_size] - recordar que el imput era de dimension [batch_size,num_steps]
        #El segundo valor retornado es el estado en la ultima capa
        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        # reshape to (batch_size * num_steps, hidden_size)
        output = tf.reshape(output, [-1, hidden_size])

        #Las variables se llaman softmax, pero aun no hemos definido el softmax. Lo hacemos mas adelante en la funcion de coste
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))
        #logits=output * softmax_w + softmax_b
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        
        # Reshape logits to be a 3-D tensor for sequence loss
        #Volvemos a hacer aparecer la dimension steps que habia "desaparecido" en el reshape que hicimos de output antes
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        #Definimos la funcion de coste
        # Use the contrib sequence loss and average over the batches
        #En target tenemos el vector con la palabra esperada
        #el tercer argumento es una matriz con pesos. Los pesos nos permitirian definir que el los en diferentes instantes de 
        #tiempo - steps -, pese diferente. En nuestro caso no lo usamos, asi que ponemos la identidad
        #average_across_timesteps determina si el coste se promediara a lo largo de los steps
        #average_across_batch=True) determina si el coste se promediara a lo largo del batch
        #En nuestro caso optamos por la tercera opcion

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.input_obj.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.cost = tf.reduce_sum(loss)

        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not is_training:
           return
        self.learning_rate = tf.Variable(0.0, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def train(train_data, vocabulary, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    # setup data and models
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocabulary,
              num_layers=num_layers)
    init_op = tf.global_variables_initializer()
    orig_decay = lr_decay
    with tf.Session() as sess:
        # start threads
        sess.run([init_op])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0.0)
            m.assign_lr(sess, learning_rate * new_lr_decay)
            # m.assign_lr(sess, learning_rate)
            # print(m.learning_rate.eval(), new_lr_decay)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))
            curr_time = dt.datetime.now()
            for step in range(training_input.epoch_size):
                # cost, _ = sess.run([m.cost, m.optimizer])
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((dt.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = dt.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print("Epoch {}, Step {}, cost: {:.3f}, accuracy: {:.3f}, Seconds per step: {:.3f}".format(epoch,
                            step, cost, acc, seconds))

            # save a model checkpoint
            saver.save(sess, data_path + '\\' + model_save_name, global_step=epoch)
        # do a final save
        saver.save(sess, data_path + '\\' + model_save_name + '-final')
        # close threads
        coord.request_stop()
        coord.join(threads)


def test(model_path, test_data, reversed_dictionary):
    test_input = Input(batch_size=20, num_steps=35, data=test_data)
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocabulary,
              num_layers=2)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # start threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))
        # restore the trained model
        saver.restore(sess, model_path)
        # get an average accuracy over num_acc_batches
        num_acc_batches = 30
        check_batch_idx = 25
        acc_check_thresh = 5
        accuracy = 0
        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true_vals, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy],
                                                               feed_dict={m.init_state: current_state})
                pred_string = [reversed_dictionary[x] for x in pred[:m.num_steps]]
                true_vals_string = [reversed_dictionary[x] for x in true_vals[0]]
                print("True values (1st line) vs predicted values (2nd line):")
                print(" ".join(true_vals_string))
                print(" ".join(pred_string))
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc
        print("Average accuracy: {:.3f}".format(accuracy / (num_acc_batches-acc_check_thresh)))
        # close threads
        coord.request_stop()
        coord.join(threads)


if args.data_path:
    data_path = args.data_path
    
train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()

if args.run_opt == 1:
    train(train_data, vocabulary, num_layers=2, num_epochs=60, batch_size=20,
          model_save_name='two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr')
else:
    trained_model = args.data_path + "\\two-layer-lstm-medium-config-60-epoch-0p93-lr-decay-10-max-lr-38"
    test(trained_model, test_data, reversed_dictionary)

