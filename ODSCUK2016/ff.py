# -*- coding: utf-8 -*-
"""
(C) 2016 Nikolay Manchev
This work is licensed under the Creative Commons Attribution 4.0 International 
License. To view a copy of this license, visit 
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

class FeedForwardNetwork(object):
  """ Feed Forward 1 Hidden Layer Neural Network 
      
      This is a single hidden layer feedforward neural network, which uses
      the sigmoid activation function. 
      
      The network uses early stopping critera and is also limited in number 
      of epoch it is allowed to run, in case a minimum wasn't found after
      a given number of iterations.
      
      We also make use of momentum in order to decrease the number of epochs
      the network takes to converge.
  """

  def __init__(self, hidden_neuron_count=4, n = 0.3, m = 0.1, max_epoch = 1000,
               validation_size = 0, verbose = False):
    """ Initialises the network
       
        Args:
            hidden_neuron_count: number of neurons in the hidden layer
            n: learning rate
            m: momentum
            max_epoch: maximum number of
            validation_size: size of the validation set in % [0.0 to 1.0]
                             Setting this parameter enables early stopping
            verbose: print epoch and error information while fitting
    """
    self.hidden_neuron_count = hidden_neuron_count
    self.n = n
    self.max_epoch = max_epoch
    self.m = m
    self.verbose = verbose
    self.validation_size = validation_size
    self.bias = 1.0

  def fit(self, i, t):
    """ Trains the network.
    
        Args:
           i: a numpy.ndarray containing the input vectors
              ex:
                [[ 4.8  3.4  1.9  0.2] # Input vector 0
                 [ 5.5  3.5  1.3  0.2] # Input vector 1
                 [ 5.2  3.5  1.5  0.2] # Input vector 2
                 ...]
           t: a numpy array containing the target classes for the input vectors
              ex:
                [0 0 2 ... ] # Target output for vector 0 is 0
                             # Target output for vector 1 is 0
                             # Target output for vector 2 is 1
    """    
    
    # If using early stopping, split the input dataset into a training
    # and validation sets
    if self.validation_size != 0:
      i_train, i_valid, t_train, t_valid = \
        cross_validation.train_test_split(i, t, 
                                          test_size=self.validation_size)
      i = i_train
      t = t_train      

    # Get the number of input vectors in the i dataset
    i_count = i.shape[0]        

    # Set input and output neuron count
    #
    # The number of input neurons is based on the number of input vector 
    # features. Ex: if i.shape is (90,4), we've got 90 input vectors, 4 
    # features each, so we need 4 neurons for the input layer.
    #    
    # Output is set according to the number of unique output classes (i.e.
    # the number of unique elements in t)
    self.input_neuron_count = i.shape[1]    
    self.output_neuron_count = len(np.unique(t))    
    
    # Initialise weights with random values
    #   w_hidden contains the weights for the hidden layer
    #   w_output contains the weights for the output layer
    # We add +1 to the neuron count to make a placeholder for the bias
    self.w_hidden = np.random.rand(self.hidden_neuron_count, 
                                   self.input_neuron_count + 1)
    self.w_output = np.random.rand(self.output_neuron_count, 
                                   self.hidden_neuron_count + 1)

    # Initialise input and output values
    # We initialise the input, hidden, and output neuron values with 0's
    self.i = np.zeros((self.input_neuron_count + 1, 1))
    self.o_hidden = np.zeros((self.hidden_neuron_count + 1, 1))
    self.o_output = np.zeros((self.output_neuron_count, 1))
    
    # Initialise input potential for hidden and output layers
    #   u_hidden is the input potential of the hidden layer
    #   u_output is the output potential of the output layer
    self.u_hidden = np.zeros((self.hidden_neuron_count, 1))
    self.u_output = np.zeros((self.output_neuron_count, 1))    
    
    # Initialise variations applied to the weights (delta W)
    #  d_hidden is the local gradient for the hidden layer
    #  d_output is the local gradient for the output layer
    self.d_hidden = np.zeros((self.hidden_neuron_count))
    self.d_output = np.zeros((self.output_neuron_count))
            
    # The *_prev variables store the previous variation values 
    # for calculating momentum.
    #   d_hidden_prev is the local gradient for the hidden layer at t-1
    #   d_output_prev is the local gradient for the output layer at t-1
    self.d_hidden_prev = np.zeros((self.hidden_neuron_count, 
                                   self.input_neuron_count + 1))
    self.d_output_prev = np.zeros((self.output_neuron_count, 
                                   self.hidden_neuron_count + 1))    

    # Counter for the current epoch
    epoch = 0
    
    # Holds the validation value from the previous run to evaluate
    # the early stopping criteria
    previous_val_err = np.finfo(np.float64).max
    
    # Keep teaching the network until the epoch limitation is reached    
    while epoch < self.max_epoch:
      err = 0.0
      # Go over the training dataset, move the network forward and
      # adjust the weights using backpropagation      
      for index in range(0,i_count):
        self._forward(i[index])
        err = err + self._backprop(t[index])
      
      # Execute on every 100th epoch
      if epoch % 100 == 0:
        
        if self.verbose:
          print ('*********************************************')
          print ('Epoch %i' % epoch)
          print ('Network accuracy on train : %-.2f' 
                % accuracy_score(t, self.predict(i)))
        #   Print validation accuracy if using early stopping
          if self.validation_size != 0:
            print ('Network accuracy on test  : %-.2f'  
                  % accuracy_score(t_valid, self.predict(i_valid)))
          
        # If we are using early stopping:
        #   * check the network error
        #   * stop the teaching process if the error is greater than
        #     the error from the previous epoch
        if self.validation_size != 0:
          val_err = self.network_error(i_valid, t_valid)
          if self.verbose:
            print ('Network error             : %-.5f' % val_err)
          if previous_val_err < val_err:
            break
          previous_val_err = val_err
        if self.verbose:
          print ('*********************************************')
      
      # Increase the epoch counter      
      epoch+=1    

    if self.verbose:
      print ("Learning complete at epoch: %i" % epoch)
    
    
  def network_error(self, i, t):
    """ Returns the mean squared error of the network.
        
        Args:
           i: input dataset
           t: target classes
           
        See fit(self, i, t) for detailed explanation on the arguments.
    """    
    net_err = mean_squared_error(t, self.predict(i))
    return net_err
    
    
  def _forward(self, i): 
    """ Forward propragation.
        This function sets the network's input to the value of i, calculates
        the neuron values based on the input potential, and set the outputs
        by using the activation function (h).
        
        Args:
          i: a single entry from the input dataset
             ex:
               [[ 4.8  3.4  1.9  0.2] # Input vector 0          
    """
    
    # Set bias and input    
    self.i[:-1, 0] = i
    self.i[-1:, 0] = self.bias     
    
    # Calculate hidden layer and set bias.
    # Input potential u_hidden is calculated by using the dot product of the
    # hidden layer's weights and the values from the network's input.
    # The results are passed to the activation function and the calculated
    # values are used as the hidden layer's output (o_hidden).
    self.u_hidden = np.dot(self.w_hidden, self.i)    
    self.o_hidden[:-1, :] = self.h(self.u_hidden)
    self.o_hidden[-1:, :] = self.bias
    
    # The values for the output layer are calculated in a similar fashion.
    self.u_output = np.dot(self.w_output, self.o_hidden)
    self.o_output = self.h(self.u_output)
    
    # The active neuron in the output layer now shows the prediction made
    # by the network for input i.
    
  def _backprop(self, target):
    """ Backpropragation.
        This function determines the deviation of the output from target,
        calculates the local gradient for the output and hidden layer,
        and changes the weights for the layers by using the network's
        learning rate and given momentum.
        
        Args:
          target: target output class for the current network state
    """    
    # Get a matrix of the difference between the target and the actual output
    # We use np.eye() to mask out the other output neurons and calculate the 
    # error only for the output relevant to the target class.
    # Ex:
    # We have three output classes. Input is:
    #  [ 0.92178306]
    #  [ 0.87316958]
    #  [ 0.75880055]
    # Target class is:
    #  [ 0.]
    #  [ 1.]
    #  [ 0.]
    # Error is:
    #  [ 0.92178306] <- unchanged
    #  [-0.12683042] <- difference from target (0.87316958 - 1.0 = -0.12683042)
    #  [ 0.75880055] <- unchanged
    err = self.o_output - np.eye(self.output_neuron_count)[int(target)].\
                                reshape(self.output_neuron_count,1) * target
    
    # Calculate local gradient as:
    #   d_o = h'(U_o) * err
    #   d_h = h'(U_h) * dot(w_o, d_o)
    self.d_output = self.h_prime(self.u_output) * err
    self.d_hidden = self.h_prime(self.u_hidden) * \
                    np.dot(self.w_output[:,:-1].T, self.d_output)

    # Calculate the changes in output and hidden by using
    # dot(local gradient, current value)
    d_dot_hidden = np.dot(self.d_hidden, self.i.T)
    d_dot_output = np.dot(self.d_output, self.o_hidden.T)

    # Apply the changes using learning rate and momentum:
    # w_h = w_h - n * hidden_weight_change - m * previous_hidden_weight_change
    # w_o = w_o - n * output_weight_change - m * previous_output_weight_change
    # where n is the learning rate and m * previous_change is term of momentum
    self.w_hidden = self.w_hidden - self.n * d_dot_hidden - \
                    self.m * self.d_hidden_prev
    self.w_output = self.w_output - self.n * d_dot_output - \
                    self.m * self.d_output_prev

    # Store the weight changes for the next calculation of momentum
    self.d_hidden_prev = d_dot_hidden
    self.d_output_prev = d_dot_output
    
    return np.mean(np.power(err,2))
    
    
  def print_state(self):
    """ Prints the network's layers for debugging purposes.
    """       
    print ("-------")
    print ("Input nueron count  :", self.input_neuron_count)
    print ("Hidden nueron count :", self.hidden_neuron_count)
    print ("Output nueron count :", self.output_neuron_count)
    print ("-------")
    print
    print ("Input i")
    print ("-------")
    print (self.i)
    print
    print ("Weights hidden layer")
    print ("-------")
    print (self.w_hidden)
    print
    print ("Input potential hidden layer")
    print ("-------")
    print (self.u_hidden)
    print
    print ("Output hidden layer")
    print ("-------")
    print (self.o_hidden)
    print
    print ("Weights output layer")
    print ("-------")
    print (self.w_output)
    print
    print ("Input potential output layer")
    print ("-------")
    print (self.u_output)
    print
    print ("Output output layer")
    print ("-------")
    print (self.o_output)
    print    

  def h(self, x):    
    """ Calculates semi-linear activiation function based on x using
        phi(x) = (1 + exp(-x))^(-1)        
    """
    return 1 / (1 + np.exp(-x))
      
  def h_prime(self, x):
    """ Returns the derivative of the activation function for calculating
        the weight updates.
        dphi(x)/dx = phi(x)*(1 - phi(x))
    """
    h = self.h(x)
    return h * (1 - h)
        
  def predict(self, i):
    """ Classifies dataset i by looping over the elements of the dataset.
        This function sets the input to the current element, rolls the network 
        forward, and takes the value of the output nodes as a prediction.
        
        Args:
           i: a numpy.ndarray containing the input vectors
              ex:
                [[ 4.8  3.4  1.9  0.2] # Input vector 0
                 [ 5.5  3.5  1.3  0.2] # Input vector 1
                 [ 5.2  3.5  1.5  0.2] # Input vector 2
                 ...]
        Output:
           Possible classes for the individual vectors in the input array.
              ex:
                [0 0 2 ... ]
    """
    # Get the number of vectors in the input dataset
    predictions_count = i.shape[0]
    # Create an empty array to contain the output classes
    predictions = np.zeros(predictions_count)
    # Loop over the individual vectors
    for index in range(0,predictions_count):
      # Set the current vector as network input and roll the network forward
      self._forward(i[index])
      
      # The number of neurons in the output layer is determined by the 
      # number of unique target classes.
      # Ex: Say we've got three unique classes and we end up with these 
      # values on the output after calling self._forward()
      # 
      # [[ 0.14294866]
      # [ 0.31053247]
      # [ 0.84564599]]
      # 
      # We use np.rint() to round them to the nearest integer (0 or 1)  to 
      # get the nuerons in boolean state (enabled or disabled). The expectation
      # is that only a single neuron should be enabled at this stage.
      #
      # [[ 0.]
      # [ 0.]
      # [ 1.]]
      #
      # We then use argmax() to get the index of the maximum values along
      # the matrix single axis. This transforms the matrix to a class number.
      # 
      # [2]
      prediction = np.rint(self.o_output).argmax(0)
      
      # Add the predicted class to the predictions array
      predictions[index] = prediction
    
    return predictions
    
