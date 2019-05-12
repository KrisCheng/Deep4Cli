#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

raw = '../../data/oni/csv/nino3_4_anomaly.csv'
series = pd.read_csv(raw, header=0, parse_dates=[0], index_col=0, squeeze=True)

raw_values = series.values
# print(raw_values)

