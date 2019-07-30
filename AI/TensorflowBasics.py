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


# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # Visualizar en tensorboard
    writer = tf.summary.FileWriter('C:/Users/Eugenio/Downloads/', sess.graph)

    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    a_out = sess.run(a)
    print("Variable a is {}".format(a_out))
    