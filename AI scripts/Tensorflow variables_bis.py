# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 07:58:01 2019

@author: Eugenio
https://jasdeep06.github.io/posts/variable-sharing-in-tensorflow/
"""

#import tensorflow
import tensorflow as tf

tf.reset_default_graph()

#open a variable scope named 'scope1'
with tf.variable_scope("scope1"):
    #add a new variable to the graph
    var1=tf.get_variable("variable1",[1])

#print the name of variable
print(var1.name)

#open a variable scope named 'scope1'
with tf.variable_scope("scope1"):
    #open a nested scope name 'scope2'
    with tf.variable_scope("scope2"):
        #add a new variable to the graph
        var2=tf.get_variable("variable2",[1])
#print the name of variable
print(var2.name)


#open a variable scope named 'scope1'
with tf.variable_scope("scope1"):
    #declare a variable named variable1
    var3 = tf.get_variable("variable3",[1])
    try:
        #declare another variable with same name
        var4 = tf.get_variable("variable3",[1])
    except:
        print("La variable ya existia, y reuse es false - valor por defecto")

#open a variable scope named 'scope1'
with tf.variable_scope("scope1"):
    #declare a variable named variable1
    var5 = tf.get_variable("variable5",[1])
	#set reuse flag to True
    tf.get_variable_scope().reuse_variables()
	#just an assertion!
    assert tf.get_variable_scope().reuse==True
	#declare another variable with same name
    var6 = tf.get_variable("variable5",[1])

assert var5==var6
#print the name of variable
print(var5.name)
#print the name of variable
print(var6.name)