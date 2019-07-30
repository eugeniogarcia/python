# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:54:47 2019

@author: Eugenio
"""

import tensorflow as tf

# To clear the defined variables and operations of the previous cell
tf.reset_default_graph()   

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")
    
# create TensorFlow variables
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, name='c')

# now create some operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')



# Create the Events DB for Tensorboard. Opcion 1. Ussing the default graph
# writer = tf.summary.FileWriter('C:/Users/Eugenio/Downloads\', tf.get_default_graph())

# Ejemplo 1 para tensorboard. Escalar
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
scalar_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)

# Ejemplo 2 para tensorboard. Histograma y Distribucion
x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
histogram_summary = tf.summary.histogram('My_histogram_summary', x_matrix)

# Ejemplo 3 para tensorboard. Imagen
w_gs = tf.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
#Formatea el tensor en 3 imagenes de 10x10 con un solo canal - escala de grises
w_gs_reshaped = tf.reshape(w_gs, (3, 10, 10, 1))
#Formatea el tensor en 5 imagenes de 10x10 con tres canales - RGB
w_c_reshaped = tf.reshape(w_c, (5, 10, 10, 3))
gs_summary = tf.summary.image('Grayscale', w_gs_reshaped)
c_summary = tf.summary.image('Color', w_c_reshaped, max_outputs=5)

# Une todos los sumarios en uno
merged = tf.summary.merge_all()


# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))

    # Create the Events DB for Tensorboard. Opcion 2. Ussing the current session graph
    writer = tf.summary.FileWriter('C:/Users/Eugenio/Downloads/', sess.graph)
    # Vamos a visualizar 100 mediciones
    for step in range(100):
        # loop over several initializations of the variable
        sess.run(init_op)
        # summary = sess.run(first_summary)
        #writer.add_summary(summary, step)
        
        #summary1, summary2 = sess.run([scalar_summary, histogram_summary])
        #writer.add_summary(summary1, step)
        #writer.add_summary(summary2, step)
        
        summary = sess.run(merged)
        writer.add_summary(summary, step)
        
    print('Done with writing the summaries')
    
