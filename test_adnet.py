#! /usr/bin/env python
# -*- coding: utf-8 -*-


import cPickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from utils import load_data
from adnet import AdversarialNets

def test_gan():

    datasets = load_data('../20150717-/mnist.pkl.gz')

    train_set, validate_set = datasets
    train_x, train_y = train_set
    validate_x, validate_y = validate_set
    xs = np.r_[train_x, validate_x]

    all_params = {
        'hyper_params': {
        },
        'optimize_params': {
        }
    }

    model = AdversarialNets(**all_params)
    model.fit(xs)

    return model


if __name__ == '__main__':
    model = test_gan()
    hist = np.vstack(model.discriminator_hist)
    plt.plot(hist[:, 0], hist[:, 1])

    size = 28
    im_size = (28, 28)
    output_image = np.zeros((size * 10, size * 10))

    for i in range(10):
        ims = model.sampling(10)
        for j, im in enumerate(ims):
            output_image[im_size[0]*i: im_size[0]*(i+1), im_size[1]*j:im_size[1]*(j+1)] = im.reshape(im_size)
    misc.imsave('sample_'+'.jpg', output_image)

    plt.show()
# End of Line.
