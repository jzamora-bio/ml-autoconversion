#!/usr/bin/env python
# coding: utf-8
# https://www.pnas.org/content/115/39/9684
# https://www.pnas.org/content/pnas/suppl/2018/09/06/1810286115.DCSupplemental/pnas.1810286115.sapp.pdf
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd 
from sklearn.preprocessing import StandardScaler
import numpy as np
import os, sys
import utils, nn

# 0.5 Config
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
resdir = utils.creadir("Results") # Crea directorio con la fecha prara guardar reultados

# 1. Reading Data
utils.pTitle("1. Reading Data")
dat = pd.read_csv("data/20200413-AutoconversionData.txt")
# dat = pd.read_csv("Data/dat.txt")
train = dat.sample(frac=0.8)
test = dat.sample(frac=0.2)
train.shape, test.shape
#! Explore
# train.hist(column='dQrauto_KCE', bins =100)


# 2. Scaling and Transform original data
utils.pTitle("2. Scaling and Transform original data")
train_val = train.values
scaler = StandardScaler()
# scaler = StandardScaler(with_mean=False)
scaler.fit(train_val)
print(scaler.mean_)
train_scal = scaler.transform(train_val)
Xtrain = train_scal[:,0:2]
Ytrain = train_scal[:,2]

test_val = test.values
test_scal = scaler.transform(test_val)
Xtest = test_scal[:,0:2]
Ytest = test_scal[:,2]
#! Explore
# plt.hist(Ytrain, bins = 100)

neuronas = [15, 45, 65, 75, 115,150]
neuronas = [75]
for i in neuronas:
	capas = [2, 3, 4]
	capas = [2]
	for c in capas:
		utils.pTitle2("Neuronas: " + str(neuronas))
		out = nn.run(Xtrain, Ytrain, Xtest, Ytest, resdir, i, c)
		#! save scaled real data
		# #predictions.T[0],Ytest
		out = (out*np.sqrt(scaler.var_[2])) + scaler.mean_[2] 
		out.to_csv(resdir + "/Predic.csv")

