#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
(C) 2015 Nikolay Manchev
This work is licensed under the Creative Commons Attribution 4.0 International 
License. To view a copy of this license, visit 
http://creativecommons.org/licenses/by/4.0/.
"""

# Common libraries
import os
import numpy as np
import time
import datetime

# Required for the k-fold analysis and for measuring accuracy
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold

# The custom feedforward network
from ff import FeedForwardNetwork

def load_iris_dataset():
  """ Loads the Iris dataset from UCI Machine Learning Repository. The data set
      contains 3 classes of 50 instances each, where each class refers to a 
      type of iris plant. One class is linearly separable from the other 2; 
      the latter are NOT linearly separable from each other.       
      
      The method returns two Numpy arrays. The first one contains the feature 
      vectors, and the second one the respective classes.
      
      If there is an invalid attribute in any record of the dataset, the
      method skips this record completely (i.e. we apply complete case
      analysis only)
  """
  
  # Location and file name for the dataset
  file_location = 'iris'
  data_file     = file_location + os.sep + 'iris.data'
  
  # We have to map the categorical class names to numbers
  mapping = { 'Iris-setosa':'0', 'Iris-versicolor':'1', 'Iris-virginica':'2'}
  
  # Prepare a feature vectors list(x) and a list to keep the classes(y)
  x = list()
  y = list()
  
  # Read the dataset line by line
  with open(data_file) as f:
    for line in f:
      # Remove end of line character
      line = line.rstrip()      
      # Map classes to numbers
      for k, v in mapping.items():
        line = line.replace(k, v)
      # Get all the line values in an array
      attr = line.split(',')  
      try:
        # Convert the values to floats and split the attributes from the class
        attr = [ float(value) for value in attr ]
        # The first four values are attributes
        x.append(attr[:4])
        # The fifth value is the respective class
        y.append(attr[-1])
      except ValueError:
        # If we can't convert all strings to floats, skip this record        
        print ('Skipping record in dataset: ', attr)
      
  # Transform the lists into Numpy arrays
  return np.array(x), np.array(y)




def cross_fold_feedforward(x, y, k=10, n=0.1, m=0.1, hidden_neuron_count=4, 
                           max_epoch=300, validation_size=0.3, verbose=False):
  print ('***************************************************')
  print ('Running a custom feedforward network                          ')
  print ('***************************************************')
  print ('Hidden layer nuron count:', hidden_neuron_count)
  print ('Learning rate:', n)
  print ('Momentum:', m)
  print ('Maximum number of epochs:', max_epoch)
  print ('Early stoping:', (validation_size != 0))
  print ('***************************************************')
  
  # Split the dataset into k folds and initialise fold counter
  kf = KFold(len(x), n_folds=k)
  fold = 1

  # Average train and test accuracy for all folds
  train_acc = 0
  test_acc = 0

  # Start the timer
  start = time.time()
  
  # Go over the folds, teaching the network on the training set and 
  # evaluate its accuracy on both training and test datasets
  for train, test in kf:
    print ('---------------------------------------------')
    print ('Running fold %i...' % fold)
    print ('---------------------------------------------')
    
    # Intialise the network and fit the training dataset
    nn = FeedForwardNetwork(hidden_neuron_count, n, m, max_epoch, 
                            validation_size, verbose)
    nn.fit(x[train], y[train])
    
    # Get the accuracy on train/test
    train_acc += accuracy_score(y[train], nn.predict(x[train]))    
    test_acc += accuracy_score(y[test], nn.predict(x[test]))

    fold += 1
    
  # Calculate and print avg. runtume and accuracy
  elapsed_time = str(datetime.timedelta(seconds=int(time.time() - start)) / k)  
  train_acc = train_acc / k
  test_acc = test_acc / k
  print ('***************************************************')
  print ('Avg. fold runtime %s.\nAccuracy on train %-.2f, accuracy on test %-.2f'\
        % (elapsed_time, train_acc, test_acc))
  print ('***************************************************')

  
# Set defaults
learning_rate  = 0.1
momentum       = 0.1
hidden_neurons = 6
epochs         = 200
verbose        = False

# Load the selected dataset
x, y = load_iris_dataset()
  
# If early stopping is enabled we set the validation sample to 30% of the
# training dataset.
validation_size = 0.3  

cross_fold_feedforward(x=x, y=y, n=learning_rate, m=momentum, 
                       hidden_neuron_count=hidden_neurons, max_epoch=
                       epochs, validation_size=validation_size, 
                       verbose=verbose)
  
