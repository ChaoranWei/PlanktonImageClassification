"""Example which shows with the MNIST dataset how Lasagne can be used."""

from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import time
#import dataload1
from imageclass import image
import warnings
warnings.filterwarnings("ignore")
import scipy.linalg.blas
import matplotlib.pyplot as plt

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

NUM_EPOCHS = 1 #input('Please enter the number of epochs: ')
BATCH_SIZE = 100 #input('Please enter the batch size: ')
NUM_HIDDEN_UNITS = 100#input('Please enter the number of hidden units: ')
LEARNING_RATE = 0.01 #input('Please enter the learning rate: ')
MOMENTUM = 0.9#input('Please enter the momentum: ')
REG = 0#input('Please enter the regularization param: ')
TRAIN = 'train.pkl'#input('Please enter the train file: ')
TEST = 'testsmall.pkl' #input('Please enter the test file: ')


def _load_data():
    """Load data from `url` and store the result in `filename`."""
#    if not os.path.exists(filename):
#        print("Downloading MNIST dataset")
#        urlretrieve(url, filename)

#    with gzip.open(filename, 'rb') as f:
#        return pickle_load(f, encoding='latin-1')
   
def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = pickle.load(open('pickle/' + TRAIN))
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    test = pickle.load(open('pickle/' + TEST))
    X_test, y_test = test
    

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=121, 
        X_testOr = X_test
    )


def build_model(input_dim, output_dim,
                batch_size=None, num_hidden_units=NUM_HIDDEN_UNITS):
    """Create a symbolic representation of a neural network with `intput_dim`
    input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
    layer.
    The training function of this model must have a mini-batch size of
    `batch_size`.
    A theano expression which represents such a network is returned.
    """
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.LeakyRectify(1/3),
        #nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        l_hidden1,
        p=0.5,
    )
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.LeakyRectify(1/3),
        #nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        l_hidden2,
        p=0.5,
    )
    l_hidden3 = lasagne.layers.DenseLayer(
            l_hidden1_dropout,
            num_units=num_hidden_units ,
            nonlinearity=lasagne.nonlinearities.LeakyRectify(0.3),
            #nonlinearity=lasagne.nonlinearities.rectify,
        )
    l_hidden3_dropout = lasagne.layers.DropoutLayer(
            l_hidden3,
            p=0.5,   
    )        
    l_out = lasagne.layers.DenseLayer(
        l_hidden3_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    
    reg = lasagne.regularization.l2(l_out)
    return l_out, reg


def create_iter_functions(dataset, output_layer,  regularization,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    output = lasagne.layers.get_output(output_layer, X_batch)
    loss_train = lasagne.objectives.categorical_crossentropy(output, y_batch) + regularization
    loss_train = loss_train.mean()

    output_test = lasagne.layers.get_output(output_layer, X_batch,
                                            deterministic=True)
    loss_eval = lasagne.objectives.categorical_crossentropy(output_test,
                                                            y_batch) + regularization
    loss_eval = loss_eval.mean()

    pred = T.argmax(output_test, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    
    #nesterov_momentum
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)
    
    #adagrad
    #updates = lasagne.updates.adagrad(
   #         loss_train, all_params, learning_rate)    
   
   #rmsprop
    #updates_rmsprop = lasagne.updates.rmsprop(
    #           loss_train, all_params, learning_rate, rho = 0.9)  
    
    #adadelta
  #  updates = lasagne.updates.adadelta(
   #             loss_train, all_params, learning_rate, rho = 0.95)     
    
    #adam method
    #updates_adam = lasagne.updates.adam(
    #                loss_train, all_params, learning_rate, beta1 = 0.9, beta2 = 0.99)     
   
    #updates = lasagne.updates.apply_nesterov_momentum(updates_rmsprop, all_params, momentum=MOMENTUM)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }


def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
    )
    iter_funcs = create_iter_functions(dataset, output_layer[0],  regularization = REG * output_layer[1],
                                       X_tensor_type=T.matrix,
                                    batch_size=BATCH_SIZE,
                                    learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    print("Starting training...")
    now = time.time()
    try:
        accuracy = []
        plt.ion()
        plt.show()
        for epoch in train(iter_funcs, dataset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            accuracy.append(epoch['valid_accuracy'])
            plt.plot(accuracy)
            plt.draw()
            #time.sleep(0.1)

            if epoch['number'] >= num_epochs:
                break
        plt.show()
        

    except KeyboardInterrupt:
        pass

    return output_layer, dataset['X_testOr']



if __name__ == '__main__':
  #  D = _load_data()    
    output_layer, test = main()
    print('load test data')
    test = pickle.load(open('pickle/test.pkl'))
    X_test, y_test = test    
    output = lasagne.layers.get_output(output_layer[0], X_test, 
                                       deterministic = True) #deterministic to turn off dropout layers
    print('evaluate test data')
    output = output.eval()
    print('get test data')
    with open('pickle/submission.pkl','w') as f:
        pickle.dump(output,f)    
#cross validation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!