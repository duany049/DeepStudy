# usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data
from cnn import *
from solver import *
from utils import *
from test import SmallDataTest

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data_dir = '../data_batches_py'
data = load_data(data_dir)
for k, v in data.items():
    print('%s: ' % k, v.shape)

testSmallData = SmallDataTest()
testSmallData.test()
