#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Peng Liu <liupeng@imscv.com>
#
# Distributed under terms of the GNU GPL3 license.

"""
This file implement a class DBN.
"""

from rbm_tf import RBM
from config import cfg


class DBN(object):

    """Docstring for DBN. """

    def __init__(self):
        """TODO: to be defined1.
        """
        self.rbm_list = []
        input_size = cfg.input_size
        for i, size in enumerate(cfg.nn_hsizes):
            self.rbm_list.append(RBM("rbm%d" % i, input_size, size))
            input_size = size

    def train(self, X):
        """TODO: Docstring for train.
        :returns: TODO

        """
        for rbm in self.rbm_list:
            rbm.train(X)
            X = rbm.rbmup(X)
