# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 11:44:14 2015

@author: kiran
"""
import theano
import theano.tensor as T
import numpy
import math
import numpy as np

def load_data(patch_filename, groundtruth_filename, valid_filename, validtruth_filename):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############


    print '... loading data'

    
    # train_array = np.load(patch_filename)
    # groundtruth_array = np.load(groundtruth_filename)


    ##################################################################
    
    
    
    train_set_x = np.load(patch_filename)
    train_set_y = np.load(groundtruth_filename)

    train_patches_length = train_set_x.shape[0]
    index = np.arange(train_patches_length)
    np.random.shuffle(index)

    print 'Training_patches: ',train_patches_length

    train_set_x = train_set_x[index[0:train_patches_length]]
    train_set_y = train_set_y[index[0:train_patches_length]]


    ##################################################################

    
    valid_patch_array = np.load(valid_filename)
    valid_truth_array = np.load(validtruth_filename)

    no_of_patches = valid_patch_array.shape[0]
    index = np.arange(no_of_patches)
    np.random.shuffle(index)
    
    
    
    n_validset = int(math.floor(0.7*no_of_patches))
    n_testset = int(math.ceil(0.3*no_of_patches))
    
    valid_set_x = valid_patch_array[index[0:n_validset]]
    valid_set_y = valid_truth_array[index[0:n_validset]]

    # valid_set_x=valid_patch_array[index[0:no_of_patches]]   #changes made on 24oct
    # valid_set_y=valid_truth_array[index[0:no_of_patches]]   #changes made on 24oct to see confusion matrix issues....

    test_set_x = valid_patch_array[index[n_validset:no_of_patches]]
    test_set_y = valid_truth_array[index[n_validset:no_of_patches]]

        
    
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
#        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    print 'before shared_dataset'
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    load_data('Training_patches.npy','training_reshape.npy','Valid_patches.npy','Valid_reshape.npy')
