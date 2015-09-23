#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

from utils import load_data
from gan import GAN

def test_gan(
        n_iters=1000,
        learning_rate=1e-4,
        n_mc_samples=1,
        scale_init=0.01,
        dim_z=2,
    ):

    datasets = load_data('../20150717-/mnist.pkl.gz')

    train_set, validate_set = datasets
    train_x, train_y = train_set
    validate_x, validate_y = validate_set
    xs = np.r_[train_x, validate_x]
    optimize_params = {
        'learning_rate' : learning_rate,
        'n_iters'       : n_iters,
        'minibatch_size': 1000,
        'calc_history'     : 'all',
        'calc_hist'     : 'all',
        'n_mod_history'    : 100,
        'n_mod_hist'    : 100,
        'patience'      : 5000,
        'patience_increase': 2,
        'improvement_threshold': 1.005,
    }

    all_params = {
            'hyper_params': {
            'rng_seed'          : 1234,
            'dim_z'             : dim_z,
            'n_hidden'          : [500, 500],
            'n_mc_sampling'     : n_mc_samples,
            'scale_init'        : scale_init,
            'nonlinear_q'       : 'relu',
            'nonlinear_p'       : 'relu',
            'type_px'           : 'bernoulli',
            'optimizer'         : 'adam',
            'learning_process'  : 'early_stopping'
        }
    }
    all_params.update({'optimize_params': optimize_params})

    model = GAN(**all_params)
    model.fit(xs)

    return datasets, model


if __name__ == '__main__':
    data, model = test_gan(
        n_iters=1000,
        learning_rate=0.001,
        n_mc_samples=1,
        scale_init=1.,
        dim_z=50,
        )
    hist = np.vstack(model.hist)
    plt.plot(hist[:, 0], hist[:, 1])

    plt.show()
# End of Line.
