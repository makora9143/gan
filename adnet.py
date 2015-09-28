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


class AdversarialNets(object):
    def __init__(
            self,
            hyper_params=None,
            optimize_params=None,
            model_params=None
        ):
        self.rng_noise = RandomStreams(1234)
        self.rng = RandomStreams(1234)
        self.dim_z = 100
        self.optimize_params = optimize_params

    def init_models(self, dim_x):

        self.discriminator_model = [
            Maxout(dim_input=dim_x, dim_output=240, piece=5, irange=0.005),
            Maxout(dim_input=240, dim_output=240, piece=5, irange=0.005),
            Layer(size=(240, 1), function='sigmoid'),
        ]

        self.discriminator_params = [
            param for layer in self.discriminator_model for param in layer.params
        ]

        self.generator_model = [
            Layer(size=(self.dim_z, 1200), function='relu'),
            Layer(size=(1200, 1200), function='relu'),
            Layer(size=(1200, dim_x), function='sigmoid'),
        ]

        self.generator_params = [
            param for layer in self.generator_model for param in layer.params
        ]

    def discriminate(self, X):
        for i, layer in enumerate(self.discriminator_model):
            if i == 0:
                output = layer.fprop(X)
                output *= 2. * (self.rng_noise.uniform(size=output.shape, dtype='float32') > .5)
            else:
                output = layer.fprop(output)
        return output

    def generate(self, Z):
        for i, layer in enumerate(self.generator_model):
            if i == 0:
                output = layer.fprop(Z)
            else:
                output = layer.fprop(output)
        return output

    def sampling(self, num_sample):
        z = np.random.uniform(
            low=-np.sqrt(3),
            high=np.sqrt(3),
            size=(num_sample, self.dim_z)
        ).astype(theano.config.floatX)
        Z = T.matrix()
        sample = theano.function(
            inputs=[Z],
            outputs=self.generate(Z)
        )
        return sample(z)

    def get_objective(self, X):
        z = self.rng_noise.uniform(
            low=-np.sqrt(3),
            high=np.sqrt(3),
            size=(X.shape[0], self.dim_z),
            dtype='float32'
        )
        generated_x = self.generate(z)
        objective_D = T.mean(T.log(self.discriminate(X)) + T.log(1 - self.discriminate(generated_x)))
        objective_G = T.mean(T.log(self.discriminate(generated_x)))

        return objective_D, objective_G

    def fit(self, x_datas):
        X = T.matrix()
        self.init_models(dim_x=x_datas.shape[1])
        objective_D, objective_G = self.get_objective(X)
        generator_gparams = T.grad(
            cost=objective_G,
            wrt=self.generator_params
        )

        discriminator_gparams = T.grad(
            cost=objective_D,
            wrt=self.discriminator_params
        )

        generator_updates = self.sgd(
            params=self.generator_params,
            gparams=generator_gparams,
            hyper_params=self.optimize_params
        )

        discriminator_updates = self.sgd(
            params=self.discriminator_params,
            gparams=discriminator_gparams,
            hyper_params=self.optimize_params
        )

        self.discriminator_hist, self.generator_hist = self.optimize(
            X=X,
            x_datas=x_datas,
            objective_G=objective_G,
            objective_D=objective_D,
            generator_updates=generator_updates,
            discriminator_updates=discriminator_updates,
        )

    def sgd(self, params, gparams, hyper_params):
        learning_rate = 0.01
        updates = OrderedDict()

        for param, gparam in zip(params, gparams):
            updates[param] = param + learning_rate * gparam

        return updates

    def momentum(self, params, gparams, hyper_params):
        updates = OrderedDict()
        learning_rate = 0.1
        momentum = shared32(0.5)
        updates[momentum] = momentum + 0.2 / 250

        for param, gparam in zip(params, gparams):
            mom = shared32(param.get_value(borrow=True) * 0.)
            mom_new = mom * momentum - learning_rate * gparam
            param_new = param - mom_new

            updates[mom] = mom_new
            updates[param] = param_new
        return updates

    def adam(self, params, gparams, hyper_params):
        updates = OrderedDict()
        decay1 = 0.1
        decay2 = 0.001
        weight_decay = 1000 / 50000.
        learning_rate = 0.01

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

            param_new = param - effstep_new

            updates[param] = param_new
            updates[mom1] = mom1_new
            updates[mom2] = mom2_new

        return updates

    def optimize(self, X, x_datas, objective_G, objective_D, generator_updates, discriminator_updates):
        train_x = x_datas
        minibatch_size = 100
        d_times = 1
        n_iters = 1000

        generator_train = theano.function(
            inputs=[X],
            outputs=objective_G,
            updates=generator_updates
        )

        discriminator_train = theano.function(
            inputs=[X],
            outputs=objective_D,
            updates=discriminator_updates
        )

        n_samples = train_x.shape[0]
        generator_hist = []
        discriminator_hist = []

        total_discremenator = []
        total_generator = []

        for i in xrange(n_iters):
            ixs = np.random.permutation(n_samples)
            for j in xrange(0, n_samples, minibatch_size):
                # for _ in xrange(d_times):
                #     discriminator_error = discriminator_train(train_x[ixs[j: j+minibatch_size]])
                discriminator_error = 0
                generator_error = generator_train(train_x[ixs[j: j+minibatch_size]])
                total_generator.append(generator_error)
                total_discremenator.append(discriminator_error)

            if i % 10 == 0:
                mean_discriminator = np.mean(total_discremenator)
                mean_generator = np.mean(total_generator)
                total_discremenator = []
                total_generator = []
                print i, 'Epoch Discremenator:', mean_discriminator, 'Generator:', mean_generator
                generator_hist.append((i, mean_generator))
                discriminator_hist.append((i, mean_discriminator))

        return discriminator_hist, generator_hist



# End of Line.
