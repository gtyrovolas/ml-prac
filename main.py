import _pickle as cp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pprint

pp = pprint.PrettyPrinter(indent=4)


# calculates the mean squared error of y and y_hat
def  meanSqErr(y, y_hat):
	return np.square(y - y_hat).mean() / 2

# makes a line graph for the two error lines
def mkLineGraph(errors, step):
	x = range(20, 600, step)
	trainErrs = [error[0] for error in errors]
	testErrs =[error[1] for error in errors]

	plt.plot(x, trainErrs, color='skyblue', linewidth=1.5, label = "Training error")
	plt.plot(x, testErrs, color='red', linewidth=1.5, label = "Testing error")
	plt.xlabel('Elements used for training')
	plt.ylabel('Error')
	plt.title('Testing vs Training error for different values of n_train')
	plt.legend()

	plt.show()
	

# constructs and shows a bar graph of the distribution of wine quality
def mkbar(y):
	qual_cnt = [0] * 10
	for y_i in y:
		qual_cnt[int(y_i)] += 1

	plt.bar(list(range(10)), qual_cnt, color = "red", log = True)
	plt.xlabel("Wine Quality")
	plt.ylabel("Number of Wines")
	plt.title("Distribution of Wine Quality")

	plt.show()

# naive predictor which returns the mean squared difference between
# the average of the training and the testing data
def naive(y_train, y_test):
	avg = np.average(y_train)
	naiveErr = meanSqErr(y_test, avg)
	return naiveErr

# regularises input data based on the training data
def regularise(X_train, X_test):
	for col, col_test in zip(X_train.T, X_test.T):
		avg = np.average(col)
		std = np.std(col)

		col -= avg
		col /= std
		col_test -= avg
		col_test /= std

# linear regression trained on trainingData and tested on testingData
def lin(trainingData, testingData):

	X_train, y_train, n_train = trainingData
	X_test, y_test, n_test = testingData

	# add a bias term
	ones_vector_train = [[1] for i in range(n_train)]
	ones_vector_test = [[1] for i in range(n_test)]
	
	# adjoin ones vector
	X_train = np.hstack((ones_vector_train, X_train))
	X_test = np.hstack((ones_vector_test, X_test))

	# calculate weights based on the formula given in the notes
	w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

	# error calculation
	trainErr = meanSqErr(y_train, X_train @ w)
	testErr = meanSqErr(y_test, X_test @ w)

	return trainErr, testErr

# splits X and y in training and testing data
def split(X, y, n_train, n_test):
	X_train = X[ : n_train]
	y_train = y[ : n_train]

	X_test = X[n_train : ]
	y_test = y[n_train : ]

	return X_train, y_train, X_test, y_test

# wrapper for regularised linear regression
def wrapLin(X, y, n_train, n_test):
	X_train, y_train, X_test, y_test = split(X.copy(), y, n_train, n_test)
	regularise(X_train, X_test)
	return lin((X_train, y_train, n_train), (X_test, y_test, n_test))


def main():

	# import white whine data, X is the input set, y is the output set
	X, y = cp.load(open('winequality-white.pickle','rb'))
	n, d = X.shape	

#	Handin 1: Bar graph
	mkbar(y)

#	Handin 2: Error for naive average "predictor" 
	n_train = int(0.8 * n)
	naiveErr = naive(y[:n_train], y[n_train:])
	print("Naive error is: " + str(naiveErr)) # naiveErr = 0.40692865000227674

#	Handin 3: Linear regression errors 
	trainErr, testErr = wrapLin(X, y, int(0.8 * n), n - int(0.8 * n))
	print("Training error is: " + str(trainErr)) # trainErr = 0.2819998086970962
	print("Testing error is:  " + str(testErr))  # testErr  = 0.2803646021141735

#	Handin 4: Overfitting graphs
	step = 20
	errors = [wrapLin(X, y, n_train, n - n_train) for n_train in range(20, 600, step)]
	mkLineGraph(errors, step)

#	The difference of training error and testing error is sufficiently small for n >= 500
#	I don't think we need more training data. Nevertheless, the model is underfitting due
#	to the expressive limitations of linear regression

main()

	



