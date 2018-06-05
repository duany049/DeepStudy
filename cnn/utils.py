# -*- coding:utf-8 -*-
import numpy as np


def rel_error(x1, x2):
    return np.max(np.abs(x1 - x2) / np.maximum(1e-8, np.abs(x1) + np.abs(x2)))
