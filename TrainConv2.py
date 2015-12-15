from __future__ import print_function

import lasagne
import theano
import theano.tensor as T
import time
import pickle
import matplotlib.pyplot as plt

from TrainConv import create_iter_functions
from TrainConv import train
import csv


NUM_EPOCHS = 1
BATCH_SIZE = 100
NUM_HIDDEN_UNITS =100
LEARNING_RATE = 0.01
MOMENTUM = 0.9 #input('Please enter the momentum: ')
REG = 0# input('Please enter the regularization param: ')
TRAIN = 'train.pkl'#input('Please enter the train file: ')
TEST = 'test.pkl'#input('Please enter the test file: ')


def load_data():
    data = pickle.load(open('pickle/' + TRAIN))
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    test = pickle.load(open('pickle/' + TEST))
    X_test, y_test = test
    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], 1, 40, 40))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, 40, 40))
    X_test = X_test.reshape((X_test.shape[0], 1, 40, 40))

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
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        output_dim=121,
        X_testOr = X_test
        )


def build_model(input_width, input_height, output_dim,
                batch_size=None):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, 1, input_width, input_height),
        )

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=16,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2, 2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=32,
        filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2, 2))
    

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    # l_hidden2 = lasagne.layers.DenseLayer(
    #     l_hidden1_dropout,
    #     num_units=256,
    #     nonlinearity=lasagne.nonlinearities.rectify,
    #     )
    # l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform(),
        )
    
    reg = 0# lasagne.regularization.l2(l_out)

    return l_out, reg


def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    output_layer = build_model(
        input_height=dataset['input_height'],
        input_width=dataset['input_width'],
        output_dim=dataset['output_dim'],
        )

    iter_funcs = create_iter_functions(
        dataset,
        output_layer[0], regularization = output_layer[1] * REG,
        X_tensor_type=T.tensor4,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, momentum=MOMENTUM
        )

    print("Starting training...")
    now = time.time()
    try:
        accuracy = []
        #plt.ion()
        #plt.show()
        for epoch in train(iter_funcs, dataset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))
            accuracy.append(epoch['valid_accuracy'])
            #plt.plot(accuracy)
            #plt.draw()
            #time.sleep(0.1)


            if epoch['number'] >= num_epochs:
                break
        #plt.show()

    except KeyboardInterrupt:
        pass

    return output_layer, dataset['X_testOr']


if __name__ == '__main__':
   
    output_layer, X_test = main()
    print('done training')
    #test = pickle.load(open('pickle/test.pkl'))
    #X_test, y_test = test    
    #print('got test data')
    for i in range(2):
        print('eval output: ' + str(i))
        output = lasagne.layers.get_output(output_layer[0], X_test[i*65200: (i+1)*65200], deterministic = True) #deterministic to turn off dropout layers
        print('output got. ')
        output = output.eval()  
        
        print('pickle output')
        with open('pickle/submission' + str(i) + '.pkl','w') as f:
            pickle.dump(output,f)        