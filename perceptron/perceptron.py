#############################################
# Author: Daniel S. Lee
# Date: 12/07/17
# Note: Perceptron implementation in Python
#############################################

# Load libraries
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(y_pred, deriv=False):
	if not deriv:
		return 1 / (1 + np.exp(y_pred))
	else:
		return np.multiply(sigmoid(y_pred),(1 - sigmoid(y_pred)))

def loss(y, y_pred):
	return np.mean(np.power(y - y_pred, 2))

def update(error, y_pred, W, deriv_func=sigmoid):
	size = len(y)
	deriv_sigmoid = deriv_func(y_pred, deriv=True)
	output_deriv = np.multiply(error, deriv_sigmoid)
	sums = [[np.mean(output_deriv * w)] for w in W]
	return np.array(sums) / size

def perceptron(X, y, alpha=0.01, iters=10):
	""" Perceptron implementation of XOR Function

		XOR Function:
		
			1) f(0, 0) = 0
			2) f(1, 1) = 0
			3) f(1, 0) = 1
			4) f(0, 1) = 1
	""" 

	# Add bias vector to X
	bias = np.matrix([[1]]*4)
	X = np.hstack((X, bias))

	# Random initialization of weights
	W = np.random.normal(loc=0, scale=1, size=(3,1))

	# Exectute Perceptron training from i=0 to i=iters
	i = 0
	
	while i < iters:
		z = np.dot(X,W) 
		y_pred = sigmoid(z)
		y_pred = np.argmax(y_pred, axis=1)
		error = loss(y, y_pred)
		W_delta = update(error, y_pred, W, deriv_func=sigmoid)
		W -= alpha * W_delta
		print('-------------- {0} ------------'.format(i))
		print('Error:', error)
		print('y_pred:', y_pred)
		print('W Delta:', W_delta)
		print('W Matrix:', W)
		i += 1
	# print(error)
	# print(y)
	# print(y_pred)
	# print(W)
	return y_pred

if __name__ == "__main__":

	X = np.matrix([[0,0],[1,1],[1,0],[0,1]])
	y = np.array([[0],[0],[1],[1]])
	y_predict = perceptron(X, y)

