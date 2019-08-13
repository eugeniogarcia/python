# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:11:32 2019

@author: Eugenio
"""
import tensorflow as tf

def add_function():
    x = tf.Variable(3, name="x_scalar")
    y = tf.Variable(2, name="y_scalar")
    addition = tf.add(x,  y, name="add_function")
    print("=== checking Variables ===")
    print("x:", x, "\ny:", y, "\n")
    return addition

#
#No reusamos variables, asi que cada llamada crea una nueva pareja de variables
#
# To check whether or noe result1 and result2 is different
# First Call creates one set of 2 variables.
result1 = add_function()
# Second Call creates another set of 2 variables.
result2 = add_function()

print("=== checking Variables ===")
print("result1:", result1, "\nresult2:", result2, "\n")


#
#Reusamos variables pasandolas como argumentos a la funcion
#
# The way to share two variables declared in one place 
variables_dict = {"x_scalar": tf.Variable(3, name="x_scalar"), 
                 "y_scalar": tf.Variable(2, name="y_scalar")}

# tf.Tensor is implicitly named like <OP_NAME>:<i>
# <OP_NAME>: the name of operation produce the tensor.
# <i>:  integer representing the index of the tensor among the operation's outputs.
def add_function_withArgs(x, y):
    addition = tf.add(x,  y, name="add_function")
    print("=== checking Variables ===")
    print("x:", x, "\ny:", y, "\n")
    return addition

# To check whether or noe result1 and result2 is the same
# First Call creates one set of 2 variables.
result3 = add_function_withArgs(variables_dict["x_scalar"], variables_dict["y_scalar"])
# Second Call also creates the same set of 2 variables.
result4 = add_function_withArgs(variables_dict["x_scalar"], variables_dict["y_scalar"])

print("=== checking Variables ===")
print("result3:", result3, "\nresult4:", result4, "\n")

# To initialize all variables in this default graph.
global_init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(global_init_op)
    print("=== checking Variables in a Session ===")
    print("result3:", result3, "\nresult4:", result4, "\n")
    result3_, result4_ = sess.run([result3, result4])
    print("=== the value each tensor has ===")
    print("result3:", result3_, "\nresult4:", result4_)
    print("[result3, result4]:", sess.run([result3, result4]))
    



