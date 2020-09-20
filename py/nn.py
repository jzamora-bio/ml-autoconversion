from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys, time
import tensorflow as tf
import utils

# Setting AND Run Neural Netowrks

def run(xtr, ytr, xte, yte, plotdir, ne, ncapas): # xtr: array features, ytr: vector output
	# Parametros
	# ne, capas, drops, epocas
	pdrop = 0.2
	epocas = 35
	alpa = 0.3

	utils.pTitle("Setting Layers")
	# dataset = tf.data.Dataset.from_tensor_slices((xtr, ytr))
	def my_leaky_relu(x):
	    return tf.nn.leaky_relu(x, alpha=alpa)
	neur = 85
	capas = []
	capas.append( tf.keras.layers.Dense(neur, activation=my_leaky_relu,input_shape=(2,)) )
	capas.append( tf.keras.layers.Dropout(pdrop) )
	# create hidden layers
	for i in range(ncapas):
		capas.append( tf.keras.layers.Dense(neur, activation=my_leaky_relu) )
		capas.append( tf.keras.layers.Dropout(pdrop) )
	capas.append( tf.keras.layers.Dense(1) )
	model = tf.keras.Sequential(capas)
	model.summary()

	utils.pTitle("Setting Optimizer")
	# opt = tf.keras.optimizers.RMSprop()	
	opt = tf.keras.optimizers.Adam()
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

	utils.pTitle("Fitting Neural Network")
	history =  model.fit(xtr, ytr, epochs=epocas, verbose = 2, 
		batch_size = 32, validation_split = 0.1)
	lastloss = history.history['loss'][-1]

	utils.pTitle("Evaluating Neural Network")
	results = model.evaluate(xte, yte)


	utils.pTitle("Make predictions")
	predictions = model.predict(xte)
	
	utils.pTitle("Plotting")
	# #predictions.flatten(),Ytest
	out = pd.DataFrame({'pred': predictions.flatten(), 'real': yte}) # haciendo la salida
	name = "neuronas-" + str(ne) + "_"	
	name = name + "layers-" + str(ncapas)  + "_"
	name = name + "droupot-" + str(pdrop)  + "_"
	name = name + "alpha-" + str(alpa) + "_"
	name = name + "epocas-" + str(epocas) + "_"
	name = name + "loss-" + "{:.4f}".format(lastloss)
	utils.pTitle2("Title Plots: " + name)
	plotear( plotdir, predictions.T[0], yte, name, log=True)
	time.sleep(2)
	plotear( plotdir, predictions.T[0], yte, name, log=False)
	len(predictions.T[0]), len(yte)
	return out


def plotear( folder, x, y, t, log=False):
	plt.figure(figsize=(17, 10), dpi=96)
	scala = 'log' # log or linear
	if log:
	    scala = '(log)'
	    plt.scatter(np.log(x), np.log(y))
	else:
	    scala = '(linear)'
	    plt.scatter(x,y)
	# plt.yscale(scala)
	# plt.xscale(scala)
	plt.xlabel('predictions ' + scala)
	plt.ylabel('real ' + scala)
	plt.grid(True)
	plt.title(t)
	plt.tight_layout() # nice trick
	# SAVING
	fecha = datetime.now().strftime("%Y%m%d-%H%M%S")
	plt.savefig(folder + '/acc-'+ fecha +'.png')
	plt.clf()


################################### END CODE








	# neur = 120
	# model = tf.keras.Sequential([
	# 	### INPUT
	#     # tf.keras.layers.Dense(40, activation='relu',input_shape=(2,)),
	#     tf.keras.layers.Dense(neur, activation=my_leaky_relu,input_shape=(2,)),
	#     tf.keras.layers.Dropout(pdrop),
	#     ### LAYER 1
	#     # tf.keras.layers.Dense(40, activation='relu'),
	#     tf.keras.layers.Dense(neur, activation=my_leaky_relu),
	#     tf.keras.layers.Dropout(pdrop),
	#     ### LAYER 2
	#     tf.keras.layers.Dense(neur, activation=my_leaky_relu),
	#     tf.keras.layers.Dropout(pdrop),    
	#     ### OUTPUT
	#     tf.keras.layers.Dense(1)
	# ])










