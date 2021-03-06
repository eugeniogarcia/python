import tensorflow as tf

data_path = "C:\\Users\Eugenio\Downloads\cifar-10-batches-bin\\"


def FIFO_queue_demo_no_coord():
    # first let's create a simple random normal Tensor to act as dummy input data
    # this operation should be run more than once, everytime the queue needs filling
    # back up.  However, it isn't in this case, because of our lack of a co-ordinator/
    # proper threading
    dummy_input = tf.random_normal([3], mean=0, stddev=1)
    # let's print so we can see when this operation is called
    dummy_input = tf.Print(dummy_input, data=[dummy_input],
                           message='New dummy inputs have been created: ', summarize=6)
    # create a FIFO queue object
    q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
    # load up the queue with our dummy input data
    enqueue_op = q.enqueue_many(dummy_input)
    # grab some data out of the queue
    data = q.dequeue()
    # create a fake graph that we can call upon
    fg = data + 1

    print("Empezando")

    # now run some operations
    with tf.Session() as sess:
        # first load up the queue
        sess.run(enqueue_op)
        # now dequeue a few times, and we should see the number of items
        # in the queue decrease
        val=sess.run(fg)
        print(val)

        val=sess.run(fg)
        print(val)
        val=sess.run(fg)
        print(val)
        # by this stage the queue will be emtpy, if we run the next time, the queue
        # will block waiting for new data
        val=sess.run(fg)
        print(val)
        # this will never print:
        print("We are here!")

def FIFO_queue_demo_with_coord():
    # first let's create a simple random normal Tensor to act as dummy input data
    # this operation should be run more than once, everytime the queue needs filling
    # back up.  However, it isn't in this case, because of our lack of a co-ordinator/
    # proper threading
    dummy_input = tf.random_normal([5], mean=0, stddev=1)
    # let's print so we can see when this operation is called
    dummy_input = tf.Print(dummy_input, data=[dummy_input],
                           message='New dummy inputs have been created: ', summarize=6)
    # create a FIFO queue object
    q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
    # load up the queue with our dummy input data
    enqueue_op = q.enqueue_many(dummy_input)

    # now setup a queue runner to handle enqueue_op outside of the main thread asynchronously
    qr = tf.train.QueueRunner(q, [enqueue_op] * 1)
    # now we need to add qr to the TensorFlow queue runners collection
    tf.train.add_queue_runner(qr)

    # grab some data out of the queue
    data = q.dequeue()
    # now print how much is left in the queue
    data = tf.Print(data, data=[q.size(), data], message='This is how many items are left in q: ')
    # create a fake graph that we can call upon
    fg = data + 1
    # now run some operations
    with tf.Session() as sess:
        # we first create a TensorFlow coordinator instance which will handle
        # all the asynchronous threads and their interactions
        coord = tf.train.Coordinator()
        # now we have to start all our queue runners - if we neglect to do this
        # the main thread will hang waiting for them to be started
        threads = tf.train.start_queue_runners(coord=coord)
        # As opposed to the previous function, we don't have to call sess.run(enqueue_op)
        # because our queue runner will figure out when this needs to be called.  It
        # will do so at the beginning, and also when the queue runs out of values

        # now dequeue a few times, and we should see the number of items
        # in the queue decrease
        val=sess.run(fg)
        print(val)
        val=sess.run(fg)
        print(val)
        val=sess.run(fg)
        print(val)
        # previously the main thread blocked / hung at this point, as it was waiting
        # for the queue to be filled.  However, it won't this time around, as we
        # now have a queue runner on another thread making sure the queue is
        # filled asynchronously
        val=sess.run(fg)
        print(val)
        val=sess.run(fg)
        print(val)
        val=sess.run(fg)
        print(val)
        # this will print, but not necessarily after the 6th call of sess.run(fg)
        # due to the asynchronous operations
        print("We are here!")

        # we have to request all threads now stop, then we can join the queue runner
        # thread back to the main thread and finish up
        coord.request_stop()
        coord.join(threads)


def cifar_shuffle_batch():
    batch_size = 128
    num_threads = 16
    
    # create a list of all our filenames
    filename_list = [data_path + 'data_batch_{}.bin'.format(i + 1) for i in range(5)]
    
    #file_q es una cola que esta inicializada con los nombres de archivo dispuestos al azar
    # create a filename queue
    #Forma custom de crear una cola que tiene una lista de strings guardada al azar
    # file_q = cifar_filename_queue(filename_list)
    #Hace lo mismo, pero usando un metodo incluido en tensorflow
    file_q = tf.train.string_input_producer(filename_list)
    
    #Obtiene una imagen de uno de los archivos de la cola. La imagen es tratada en este metodo
    # read the data - this contains a FixedLengthRecordReader object which handles the
    # de-queueing of the files.  It returns a processed image and label, with shapes
    # ready for a convolutional neural network
    image, label = read_data(file_q)
    
    # setup minimum number of examples that can remain in the queue after dequeuing before blocking
    # occurs (i.e. enqueuing is forced) - the higher the number the better the mixing but
    # longer initial load time
    min_after_dequeue = 10000
    # setup the capacity of the queue - this is based on recommendations by TensorFlow to ensure
    # good mixing
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    
    #Metodo custom que se encarga de obtener una batch de imagenes
    # image_batch, label_batch = cifar_shuffle_queue_batch(image, label, batch_size, num_threads)
    #Los datos de entrada
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue,
                                                      num_threads=num_threads)
    #image_batch, label_batch = cifar_shuffle_queue_batch(image, label, batch_size, capacity, min_after_dequeue, num_threads)
    # now run the training
    cifar_run(image_batch, label_batch)


def cifar_run(image, label):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            image_batch, label_batch = sess.run([image, label])
            print(image_batch.shape, label_batch.shape)

        coord.request_stop()
        coord.join(threads)


#Metodo custom que crea una cola con una serie de valores strings tomados desde una lista, y guardados
#al azar
def cifar_filename_queue(filename_list):
    # convert the list to a tensor
    string_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
    
    # randomize the tensor
    tf.random_shuffle(string_tensor)
    
    # create the queue
    fq = tf.FIFOQueue(capacity=10, dtypes=tf.string)
    
    # create our enqueue_op for this q
    fq_enqueue_op = fq.enqueue_many([string_tensor])
    
    # create a QueueRunner and add to queue runner list
    # we only need one thread for this simple queue
    tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_enqueue_op] * 1))
    
    return fq

#Metodo custom que se encarga de crear una cola que procesa sus items al azar
# La cola tiene una capavidad y un tamaño minimo que fijamos, y se asocia a un runner
#con un un numero de threads
def cifar_shuffle_queue_batch(image, label, batch_size, capacity, min_after_dequeue, threads):
    tensor_list = [image, label]
    
    dtypes = [tf.float32, tf.int32]
    shapes = [image.get_shape(), label.get_shape()]

    #Crea la cola    
    q = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue,
                              dtypes=dtypes, shapes=shapes)
    
    #Añade a la cola los datos, con una lista de tensores
    enqueue_op = q.enqueue(tensor_list)
    
    # add to the queue runner
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * threads))
    
    #Extraemos de la cola tantos valores como indiquemos en el batch size
    # now extract the batch
    image_batch, label_batch = q.dequeue_many(batch_size)

    return image_batch, label_batch


#Metodo custom que lee los contenidos de un archivo
#El archivo es un binario que contiene una imagen
def read_data(file_q):
    #Define el obejeto "hueco", sin implementar
    # Code from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
    class CIFAR10Record(object):
        pass

    #Crea una instancia del objeto
    result = CIFAR10Record()

    #Crea una serie de propieades constantes en el objeto
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    #Tamaño de la etiqueta
    label_bytes = 1  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    
    #Tamaño de la imagen
    image_bytes = result.height * result.width * result.depth
    
    #Calculamos el tamaño de cada registro en el archivo. Cada imagen tiene un tamaño fijo
    #Aqui calculamos ese tamaño fijo
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    #Este es un reader que se incluye con tensorflow para leer precisamente 
    #registros de un tamaño fijo. Creamos el lector indicando el tamño del registro
    #El constructor es como sigue:
    #header_bytes=None,
    #    footer_bytes=None,
    #    hop_bytes=None,
    #    name=None,
    #    encoding=None
    #
    #Como bo hemos especificado se asume que el registro no tiene cabecera ni pie,
    # y que no hay hop

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    
    #El metodo read acepta como argumento una cola. Se encarga de hacer el dequeue, tomar
    #como nombre de archivo lo que ha obtenido de la cola,
    #Leer el archivo y hacer el parsing, retornando el contenido del registro y una key
    # read devuleve el siguiente registro en el formato (key, value), donde
    #value son los datos del registro.
    
    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    result.key, value = reader.read(file_q)

    #Convert raw byte strings into tensors
    #Comvertimos cada byte que hemos leido a un uint8
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    #recortamos desde la posicion inicial, label_bytes, y convertimos a int32. Esto sera la etiqueta
    #de nuestra imagen
    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image
    # we reshape the image from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    #Y finalmente tenemos nuestra imagen
    reshaped_image = tf.cast(result.uint8image, tf.float32)

    height = 24
    width = 24

    #Procedemos a transformar la imagen. Esta es la normalizacion previa a utilizar la 
    #imagen con nuestra CNN
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    result.label.set_shape([1])

    return float_image, result.label

if __name__ == "__main__":
    run_opt = 3
    if run_opt == 1:
        FIFO_queue_demo_no_coord()
    elif run_opt == 2:
        FIFO_queue_demo_with_coord()
    elif run_opt == 3:
        cifar_shuffle_batch()



