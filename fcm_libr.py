import tensorflow as tf
from tensorflow import keras
import numpy as np
from  matplotlib import pyplot as plt
import pandas as pd 
import time


class fcm:
    def __init__(self,concepts_tensor, weights_tensor, n_iterations = 30, equilibrium_point=0.001):
        ''' 
            Initialize an FCM with tensorflow
            Provide tensors of concepts and weights,
            tensors must be 1 x N and N x N 
            n_iterations: maximum iterations
            equilibrium_point: stopping condition 
        ''' 
        self.concepts_tensor = concepts_tensor
        self.weights_tensor = weights_tensor
        self.n_iterations = n_iterations -1 #due to indexing of zero
        self.equilibrium_point = equilibrium_point

        if not tf.is_tensor(self.concepts_tensor):
            self.concepts_tensor = tf.convert_to_tensor(self.concepts_tensor)

        if not tf.is_tensor(self.weights_tensor):
            self.weights_tensor = tf.convert_to_tensor(self.weights_tensor)

        assert len(self.concepts_tensor.shape.as_list()) == len(self.weights_tensor.shape.as_list())
        if self.concepts_tensor.dtype != self.weights_tensor.dtype:
            self.concepts_tensor = tf.cast(self.concepts_tensor, self.weights_tensor.dtype)
            
        

    
    def inference(self, transfer_function, inference_rule = 'stylios', verbose = -1, classification = False, output_concepts=None, concepts_activation = 'output'):
        '''
        transfer_function: one of the sigmoid,bivalent, trivalent
        inference(lambda x: sigmoid(x,l= 1), inference_rule...) #change the l factor
        inference_rule: one of 'stylios','kosko','rescaled'. (default 'stylios') 
        verbose: printing the learning proccess

        classification: Either the inference is performed for classification purposes
        output_concepts: To be specified if classification is True
        concepts_activation: One of 'output', 'all'. When 'output' is given, it only the output concept is activated


        Returns the convergence (last iteration)
        '''
        iteration = 0
        stopping_conditions = False
        self.classification = classification
        self.out_concepts = output_concepts
        self.inference_process = {} 
        self.inference_process[iteration] = self.concepts_tensor
        self.start = time.time()
        self.times = []
        self.times.append(self.start)
        while iteration <= self.n_iterations and not stopping_conditions:
            if self.classification:
              inference_output = self._calculate_classification(self.inference_process[iteration], transfer_function, inference_rule, concepts_activation)
            else:
              inference_output = self._calculate(self.inference_process[iteration], transfer_function, inference_rule)
            iteration += 1
            self.inference_process[iteration] = inference_output
            stopping_conditions = self._check_stopping_conditions( iteration)
            self._print_progress(verbose,iteration,stopping_conditions)
        return self.inference_process[iteration]
    

    def _check_stopping_conditions(self, c_iteration):
        ''' Updates the stopping criteria
            learning_dictionairy: learning is stored as dictionairy with {iter:output} pairs
            c_iteration: the current iteration
        '''
        #currently looking only for equilibrium
        #to do check for cycles 
        #to do more stopping conditions
        p_iteration = c_iteration - 1 
        difference = tf.math.reduce_max(tf.math.abs(self.inference_process[c_iteration] - self.inference_process[p_iteration]))
        if difference > self.equilibrium_point:
            return False
        else:
            return True

    def _calculate(self,input_values, transfer_function, inference_rule):
        if inference_rule == 'kosko':
            output_values = self._kosko(input_values)
        elif inference_rule == 'rescaled':
            output_values = self._rescaled(input_values)
        else:
            output_values = self._stylios(input_values)
        if self.classification:
          #convert to numpy to perform indexing substitution
          output_numpy = output_values.numpy().copy()
          output_numpy[:,-self.out_concepts:] = transfer_function(output_numpy[:,-self.out_concepts:]).numpy()
          output_numpy[:,:-self.out_concepts] = input_values.numpy()[:,:-self.out_concepts]
          output_values = tf.convert_to_tensor(output_numpy)
        else:
          output_values = transfer_function(output_values)
        return output_values

    def _calculate_classification(self,input_values, transfer_function, inference_rule, concepts_activation):
        if inference_rule == 'kosko':
            output_values = self._kosko(input_values)
        elif inference_rule == 'rescaled':
            output_values = self._rescaled(input_values)
        else:
            output_values = self._stylios(input_values)
        if concepts_activation == 'output':
          #convert to numpy to perform indexing substitution
          output_numpy = output_values.numpy().copy()
          output_numpy[:,-self.out_concepts:] = transfer_function(output_numpy[:,-self.out_concepts:]).numpy()
          output_numpy[:,:-self.out_concepts] = input_values.numpy()[:,:-self.out_concepts]
          output_values = tf.convert_to_tensor(output_numpy)
        else:
          output_values = transfer_function(output_values)
        return output_values

    def _kosko(self,c_tensor):
        return tf.linalg.matmul(c_tensor, self.weights_tensor)


    def _stylios(self,c_tensor):
        return c_tensor + tf.linalg.matmul(c_tensor, self.weights_tensor)

    def _rescaled(self,c_tensor):
        return (2*c_tensor -1) + tf.linalg.matmul((2*c_tensor -1), self.weights_tensor)

    def _print_progress(self, verbose, iteration, stopping_conditions):
        self.stop = time.time()
        self.ex_time = self.stop-self.times[0]
        self.iter_time = self.stop-self.times[-1]
        self.times.append(self.stop)

        if iteration == 1 and verbose != -1:
            print('learning started...')
        if verbose == 0 and (stopping_conditions or iteration == self.n_iterations):
            print('learning stopped. Execution time = {} s'.format(np.round(self.ex_time,4)))
        elif verbose == 1 and not (stopping_conditions or iteration == self.n_iterations):
            print('FCM iteration {}, Execution time = {} ms'.format(iteration, np.round(self.iter_time*100,3)))
        elif verbose == 1 and (stopping_conditions or iteration == self.n_iterations):
            print('learning stopped. Execution time = {} s'.format(np.round(self.ex_time,4)))




def sigmoid(x,l = 1):
    return 1/(1 + tf.math.exp(-x *l))

def trivalent(x):
    x = x.numpy()
    x[x>0] = 1
    x[x==0] = 0
    x[x<0] = -1
    return tf.convert_to_tensor(x)
    

def bivalent(x):
    x = x.numpy()
    x[x>0] = 1
    x[x<=0] = 0
    return tf.convert_to_tensor(x)
    
def softmax(x):
  return tf.nn.softmax(x)

def tanh(x):
  return tf.keras.activations.tanh(x)
