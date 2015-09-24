#! /usr/bin/env python
# -*- coding: utf-8 -*-


from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.cross_validation import train_test_split

from mlp import Layer, Maxout


def shared32(x):
    return theano.shared(np.asarray(x).astype(theano.config.floatX))


class GAN(object):
    def __init__(
        self,
        hyper_params=None,
        optimize_params=None,
        model_params=None,
        model_name=None):

        self.model_name = model_name

        self.hyper_params = hyper_params
        self.optimize_params = optimize_params
        self.model_params = model_params

        self.rng = np.random.RandomState(hyper_params['rng_seed'])

        self.model_params_ = None
        self.decode_main = None
        self.encode_main = None

    def relu(self, x): return x*(x>0) + 0.01 * x
    def softplus(self, x): return T.log(T.exp(x) + 1)
    def identify(self, x): return x
    def get_name(self): return self.model_name

    def init_model_params(self, dim_x):
        # dim_z = self.hyper_params['dim_z']
        dim_z = 1200

        activation = {
            'tanh': T.tanh,
            'relu': self.relu,
            'softplus': self.softplus,
            'sigmoid': T.nnet.sigmoid,
            'none': self.identify,
        }

        self.generator_model = [
            Layer(param_shape=(dim_z, 1200), function=activation['relu']),
            Layer(param_shape=(1200, 1200), function=activation['relu']),
            Layer(param_shape=(1200, dim_x), function=activation['sigmoid']),
        ]

        self.generator_model_params = [
            param for layer in self.generator_model
            for param in layer.params
        ]

        self.discriminator_model = [
            Maxout(dim_input=dim_x, dim_output=240, piece=2),
            Maxout(dim_input=240, dim_output=240, piece=2),
            Layer(param_shape=(240, 1), function=activation['sigmoid'])
        ]

        self.discriminator_model_params = [
            param for layer in self.discriminator_model
            for param in layer.params
        ]

    def generate_x(self, Z):
        for i, layer in enumerate(self.generator_model):
            if i == 0:
                layer_out = layer.fprop(Z)
            else:
                layer_out = layer.fprop(layer_out)

        return layer_out

    def discriminate_x(self, X):
        for i, layer in enumerate(self.discriminator_model):
            if i == 0:
                layer_out = layer.fprop(X)
            else:
                layer_out = layer.fprop(layer_out)

        return layer_out

    def get_cost_function(self, X):
        z = self.rng_noise.uniform(size=(X.shape[0], 1200))
        x_tilda = self.generate_x(z)
        return (
            T.sum(T.log(self.discriminate_x(X))) / X.shape[0],
            T.sum(T.log(1 - self.discriminate_x(x_tilda))) / X.shape[0]
        )

    def create_fake_x(self, num_sample):
        # z = self.rng_noise.uniform(size=(num_sample, 1200))
        z = np.random.uniform(size=(num_sample, 1200)).astype(theano.config.floatX)
        Z = T.matrix()
        create = theano.function(
            inputs=[Z],
            outputs=self.generate_x(Z)
        )
        return create(z)

    def fit(self, x_datas):
        X = T.matrix()
        self.rng_noise = RandomStreams(self.hyper_params['rng_seed'])

        self.init_model_params(dim_x=x_datas.shape[1])

        dist_cost, dist_gen_cost = self.get_cost_function(X)

        # gradient discriminator
        dist_gparms = T.grad(
            cost=dist_cost+dist_gen_cost,
            wrt=self.discriminator_model_params
        )

        # gradient generator
        gen_gparams = T.grad(
            cost=dist_gen_cost,
            wrt=self.generator_model_params
        )

        discriminator_updates = self.adam(self.discriminator_model_params, dist_gparms, self.optimize_params, minimum=False)
        generate_updates = self.adam(self.generator_model_params, gen_gparams, self.optimize_params)

        self.hist = self.optimize(
            X,
            x_datas,
            self.optimize_params,
            dist_cost,
            dist_gen_cost,
            discriminator_updates,
            generate_updates,
            self.rng,
        )

    def sgd(self, params, gparams, hyper_params, minimum=True):
        learning_rate = hyper_params['learning_rate']
        updates = OrderedDict()

        for param, gparam in zip(params, gparams):
            if minimum:
                updates[param] = param - learning_rate * gparam
            else:
                updates[param] = param + learning_rate * gparam

        return updates

    def adam(self, params, gparams, hyper_params, minimum=True):
        updates = OrderedDict()
        decay1 = 0.1
        decay2 = 0.001
        weight_decay = 1000 / 50000.
        learning_rate = hyper_params['learning_rate']

        it = shared32(0.)
        updates[it] = it + 1.

        fix1 = 1. - (1. - decay1) ** (it + 1.)
        fix2 = 1. - (1. - decay2) ** (it + 1.)

        lr_t = learning_rate * T.sqrt(fix2) / fix1

        for param, gparam in zip(params, gparams):
            if weight_decay > 0:
                gparam -= weight_decay * param

            mom1 = shared32(param.get_value(borrow=True) * 0.)
            mom2 = shared32(param.get_value(borrow=True) * 0.)

            mom1_new = mom1 + decay1 * (gparam - mom1)
            mom2_new = mom2 + decay2 * (T.sqr(gparam) - mom2)

            effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-10)

            effstep_new = lr_t * effgrad

            if minimum:
                param_new = param - effstep_new
            else:
                param_new = param + effstep_new

            updates[param] = param_new
            updates[mom1] = mom1_new
            updates[mom2] = mom2_new

        return updates

    def optimize(self, X, x_datas, optimize_params, dist_cost, gen_cost, dist_updates, gen_updates, rng):
        n_iters = optimize_params['n_iters']
        minibatch_size = optimize_params['minibatch_size']
        n_mod_history = optimize_params['n_mod_history']

        train_x, valid_x = train_test_split(x_datas, train_size=5./6)

        train_discrimenator = theano.function(
            inputs=[X],
            outputs=dist_cost,
            updates=dist_updates
        )

        train_generator = theano.function(
            inputs=[X],
            outputs=gen_cost,
            updates=gen_updates
        )

        valid = theano.function(
            inputs=[X],
            outputs=[dist_cost, gen_cost]
        )

        n_samples = train_x.shape[0]
        cost_history = []

        total_cost = 0
        total_gen = 0
        total_dist = 0
        num = n_samples / minibatch_size

        print 'start learning'
        for i in xrange(n_iters):
            ixs = rng.permutation(n_samples)
            for j in xrange(0, n_samples, minibatch_size):
                dist_cost = train_discrimenator(train_x[ixs[j:j+minibatch_size]])
                gen_cost = train_generator(train_x[ixs[j:j+minibatch_size]])
                total_cost += (dist_cost + gen_cost)
                total_gen += gen_cost
                total_dist = dist_cost

            if np.mod(i, n_mod_history) == 0:
                print ('%d epoch train discriminator error: %.3f, generator error: %.3f, total error: %.3f' %
                      (i, total_dist / num, total_gen / num, total_cost / num))
                valid_dist, valid_gen = valid(valid_x)
                print ('\tvalid Discriminator error: %.3f, Generator error: %.3f, total error: %.3f' %
                       (valid_dist, valid_gen, valid_dist+valid_gen))
                cost_history.append((i, valid_dist+valid_gen))
        return cost_history

    def early_stopping(self, X, x_datas, optimize_params, dist_cost, gen_cost, dist_updates, gen_updates, rng):

        train_discrimenator = theano.function(
            inputs=[X],
            outputs=dist_cost,
            updates=dist_updates
        )

        train_generator = theano.function(
            inputs=[X],
            outputs=gen_cost,
            updates=gen_updates
        )

        valid = theano.function(
            inputs=[X],
            outputs=dist_cost+gen_cost
        )

        patience = optimize_params['patience']
        patience_increase = optimize_params['patience_increase']
        improvement_threshold = optimize_params['improvement_threshold']
        minibatch_size = optimize_params['minibatch_size']

        train_x, valid_x = train_test_split(x_datas, train_size=5./6)

        n_samples = train_x.shape[0]
        n_minibatches = n_samples / minibatch_size
        cost_history = []
        best_params = None
        valid_best_error = - np.inf
        best_epoch = 0

        done_looping = False

        for i in xrange(1000000):
            if done_looping: break
            ixs = rng.permutation(n_samples)
            for j in xrange(0, n_samples, minibatch_size):
                dist_cost = train_discrimenator(train_x[ixs[j:j+minibatch_size]])
                gen_cost = train_generator(train_x[ixs[j:j+minibatch_size]])

                iter = i * (n_minibatches) + j / minibatch_size

                if (iter+1) % 50 == 0:
                    valid_error = 0.
                    for _ in xrange(3):
                        valid_error += valid(valid_x)
                    valid_error /= 3
                    if i % 100 == 0:
                        print ('epoch %d, minibatch %d/%d, valid total error: %.3f' %
                               (i, j / minibatch_size + 1, n_samples / minibatch_size, valid_error))
                    cost_history.append((i*j, valid_error))
                    if valid_error > valid_best_error:
                        if valid_error > valid_best_error * improvement_threshold:
                            patience = max(patience, iter * patience_increase)
                        best_params = self.model_params_
                        valid_best_error = valid_error
                        best_epoch = i

                if patience <= iter:
                    done_looping = True
                    break
        print ('epoch %d, minibatch %d/%d, valid best error: %.3f' %
               (best_epoch, j / minibatch_size + 1, n_samples / minibatch_size, valid_best_error))
        self.model_params_ = best_params
        return cost_history

# End of Line.
