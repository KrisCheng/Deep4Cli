#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Desc:  the file for preprocess netcdf file of ENSO
Author: Kris Peng
Copyright (c) 2017 - Kris Peng <kris.dacpc@gmail.com>
'''

from netCDF4 import Dataset

data = Dataset("../data/sst/sst.mon.mean.nc")

# the basic information
# print(data.data_model)
# print(data.dimensions.keys())
# print(data.dimensions['time'])
# print(data.variables['sst'])

sst = data.variables['sst']
print(sst)
# print(sst[1][:][:])



data.close()
