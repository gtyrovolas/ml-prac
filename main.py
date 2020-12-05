import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# calculates the mean squared error of y and y_hat
def  meanSqErr(y, y_hat):
	return np.square(y - y_hat).mean() / 2

# constructs and shows a bar graph for the distribution of wine quality
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
# the average and the outputs
def naive(y_train, y_test):
	avg = np.average(y_train)
	naiveErr = meanSqErr(y_test, avg)
	return naiveErr

# function used to regularise input data based on the training data
def regularise(X_train, X_test):
	for col, col_test in zip(X_train.T, X_test.T):
		avg = np.average(col)
		std = np.std(col)

		col -= avg
		col /= std
		col_test -= avg
		col_test /= std

# linear regression trained on X_train and tested on X_test
def lin(trainingData, testingData):

	X_train, y_train, n_train = trainingData
	X_test, y_test, n_test = testingData

	# regularise training and testing data
	regularise(X_train, X_test)

	# add a bias term
	ones_vector_train = [[1] for i in range(n_train)]
	ones_vector_test = [[1] for i in range(n_test)]
	
	X_train = np.hstack((ones_vector_train, X_train))
	X_test = np.hstack((ones_vector_test, X_test))

	# based on the formula for w given in the notes
	w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

	# error calculation
	trainErr = meanSqErr(y_train, X_train @ w)
	testErr = meanSqErr(y_test, X_test @ w)

	return trainErr, testErr

def split(X, y, n_train, n_test):

	X_train = X[ : n_train]
	y_train = y[ : n_train]

	X_test = X[n_train : ]
	y_test = y[n_train : ]

	return X_train, y_train, X_test, y_test

def main():

	# import white whine data, X input set, y is the output set
	X, y = cp.load(open('winequality-white.pickle','rb'))
	
#	mkbar(y)

	# split up  the data
	n, d = X.shape

	n_train = int(n * 0.8)
	n_test = n - n_train

	# pass a copy of X so the regularisation phase doesn't affect X
	X_train, y_train, X_test, y_test = split(X.copy(), y, n_train, n_test)

	naiveErr = naive(y_train, y_test)
	linTrainErr, linTestErr = lin((X_train, y_train, n_train), (X_test, y_test, n_test))

	print("The mean squared difference between the average and the outputs is: " + str(naiveErr))
	print("Linear regression training error is: " + str(linTrainErr) + 
		" testing error is: " + str(linTestErr))
	# 0.40692865000227674
	# 0.2819998086970962 0.2803646021141735



main()

	



