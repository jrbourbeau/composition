#!/usr/bin/env python

from __future__ import division
import numpy as np
import pandas as pd

def ratio_error(num, num_err, den, den_err):
    ratio = num/den
    ratio_err = ratio * np.sqrt((num_err / num)**2 + (den_err / den)**2)
    return ratio, ratio_err
