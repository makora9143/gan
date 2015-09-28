#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import theano
import theano.tensor as T


RNG = np.random.RandomState(1234)


def U(size, irange=None, rng=RNG):
    if irange:
        return rng.uniform(
                low=-irange,
                high=irange,
                size=size)

    else:
        return rng.uniform(
                low=-np.sqrt(6. / np.sum(size)),
                high=np.sqrt(6. / np.sum(size)),
                size=size)


def sharedX(x):
    return theano.shared(np.asarray(x).astype(theano.config.floatX))


def relu(x): return x*(x>0) + 0.01 * x
def softplus(x): return T.log(T.exp(x) + 1)
def identify(x): return x


ACTIVATION = {
    'tanh': T.tanh,
    'relu': relu,
    'softplus': softplus,
    'sigmoid': T.nnet.sigmoid,
    'none': identify,
}


class Layer(object):
    def __init__(self, size,
                 rng=RNG,
                 function='tanh',
                 w_zero=False, b_zero=False,
                 nonbias=False,
                 irange=None):

        self.size= size
        self.nonbias = nonbias

        self.W = sharedX(U(size, irange=irange))
        if w_zero:
            self.W = sharedX(np.zeros(size))

        self.params = [self.W]

        if not self.nonbias:
            if b_zero:
                self.b = sharedX(np.zeros((size[1],)))
            else:
                self.b = sharedX(U((size[1],), irange=irange))
            self.params.append(self.b)

        self.function = ACTIVATION[function]

    def fprop(self, x):
        if not self.nonbias:
            return self.function(T.dot(x, self.W) + self.b)
        else:
            return self.function(T.dot(x, self.W))


class Maxout(object):
    def __init__(self, dim_input, dim_output, piece=2, irange=None, rng=RNG):
        self.size= (piece, dim_input, dim_output)

        self.W = sharedX(U(self.size, irange=irange))
        self.b = sharedX(U((piece, dim_output), irange=irange))

        self.params = [self.W, self.b]

    def fprop(self, x):
        return T.max(T.dot(x, self.W) + self.b, axis=1)


# End of Line.
