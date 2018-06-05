# - * - coding:utf-8 -*-
import numpy as np


def sgd(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    w -= config['learning_rate'] * dw
    return w, config


def sgd_moment(w, dw, config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('moment', 0.9)
    v = config.get('velocity', np.zeros_like(w))
    v = config['moment'] * v - config['learning_rate'] * dw
    w = w + v
    config['velocity'] = v
    return w, config
