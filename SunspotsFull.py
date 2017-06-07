

# http://github.com/timestocome

# read in sunspot data and predict sunspots
# http://www.sidc.be/silso/datafiles
# http://surface.syr.edu/cgi/viewcontent.cgi?article=1056&context=eecs_techreports


# Finished:
# forward feed

# Todo:
# regularization
# data normalization
# cost function




#              I
#            /   \
#        -> H <-> H <-
#            \   /
#              O
# Input to each of the hidden
# Each hidden to itself
# Each hidden to each neighbor left - don't wrap
# Each hidden to each neighbor right - don't wrap
# Sum all hiddens to output


import numpy as np 
import pandas as pd
import theano 
from theano import function
import theano.tensor as T
import matplotlib.pyplot as plt 



# set up 
rng = np.random.RandomState(27)

# setup theano
GPU = True
if GPU:
    print("Device set to GPU")
    try: theano.config.device = 'gpu'
    except: pass    # its already set
    theano.config.floatX = 'float32'
else:
    print("Running with CPU")





def read_data_file():

    data = pd.read_csv('SN_d_tot_V2.0.csv', sep=';', header=None)
    data.columns = ['year', 'month', 'day', 'FracDate', 'Sunspots', 'Std', 'Observations', 'Verified']

    # keep only what's necessary
    data = data[['year', 'month', 'day', 'Sunspots']]

    # convert month, day, year to date index
    data['Date'] = pd.to_datetime(data[['year','month','day']])
    data = data.set_index(['Date'])

    # drop month, day, year now that date is combined
    data = data[['Sunspots']]

    # x is today's data, shift by 1 so y is tomorrow's data
    x = data['Sunspots'].values
    y = data['Sunspots'].shift(1).values

    # remove as many items as we've shifted from top of array
    x = np.delete(x, 0)
    y = np.delete(y, 0)

    x = np.asarray(x.reshape(len(x)))
    y = np.asarray(y.reshape(len(x)))

    return x.astype('float32'), y.astype('float32')

x, y = read_data_file()

######################################################################
# network constants
#######################################################################

learning_rate = 0.001
epochs = 2
n_samples = len(x)
n_hidden = 11
n_in = 1
n_out = 1


########################################################################
# test
#######################################################################


class FullyConnected:

    def __init__(self):
        
        # set up and initialize weights
        W_in_values = np.random.rand(n_hidden)
        self.W_in = theano.shared(value=W_in_values, name='W_in', borrow=True)

        W_h_values = np.random.rand(n_hidden, n_hidden)
        self.W_h = theano.shared(value=W_h_values, name='W_h', borrow=True)

        W_out_values = np.random.rand(n_hidden)
        self.W_out = theano.shared(value=W_out_values, name='W_out', borrow=True)


        self.parameters = [self.W_in, self.W_h, self.W_out]


        def save_weights():
            np.savez("Sunspot_weights.npz", *[p.get_value() for p in self.parameters])
        self.save_weights = save_weights
   

        # placeholders for data
        X = T.dscalar('X')
        Y = T.dscalar('Y')

        # -------------  feed forward ----------------------------

        # feed input to hidden units
        hidden_units = X * self.W_in 

        # take hidden output and send to other hidden nodes
        hidden_hidden = hidden_units * self.W_h      # node by column of weights

        # input from other hidden nodes
        hidden_out = T.nnet.relu(hidden_hidden.sum(axis=1) ) # sum row of weights

        # out from hidden nodes to output weights
        out = T.nnet.relu(hidden_out * self.W_out)   # hidden node outputs * output weights

        # predicted
        predicted = T.sum(out)                  # sum all incoming 

        # error 
        cost = (predicted - Y)

         
        gradients = T.grad(cost, self.parameters)    # derivatives
        updates = [(p, p - learning_rate * g) for p, g in zip(self.parameters, gradients)]


        # training and prediction functions
        self.predict_op = theano.function(inputs = [X], outputs = predicted)

        self.train_op = theano.function(
                    inputs = [X, Y],
                    outputs = cost,
                    updates = updates
        )


    def train(self, x, y):

        costs = []
        for i in range(epochs):
            
            cost = 0
            for j in range(len(y)):
                c = self.train_op(x[j], y[j])
                cost += c 

                p = self.predict_op(x[j])
                print("x: ", x[j], "y: ", y[j], "predicted: ", p, "cost: ", c)
            
            # output cost so user can see training progress
            cost /= len(y)
            print ("i:", i, "cost:", cost, "%")
            costs.append(cost)
            
        # graph to show accuracy progress - cost function should decrease
        #plt.plot(costs)
        #plt.show()


network = FullyConnected()
network.train(x, y)

