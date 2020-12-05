import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# makes a bar graph showing the distribution of wine quality
def mkbar():
	qual_cnt = [0] * 10
	for y_i in y:
		qual_cnt[int(y_i)] += 1
	print(qual_cnt)

	plt.bar(list(range(10)), qual_cnt, color = "red", log = True)
	plt.xlabel("Wine Quality")
	plt.ylabel("Number of Wines")
	plt.title("Distribution of Wine Quality")

	plt.show()

# naive predictor which returns the mean squared difference between
# the average and the outputs
def naive(y_train, y_test):
	pred = np.average(y_train)
	naiveErr = np.square(y_test - pred).mean() / 2

	return naiveErr

def errCalc(X, y, w):
	return np.square(y - X @ w).mean() / 2

def lin(X_train, y_train, X_test, y_test):

	for col, col_test in zip(X_train.T, X_test.T):
		avg = np.average(col)
		std = np.std(col)

		col -= avg
		col /= std
		col_test -= avg
		col_test /= std

	ones_vector = [[1] for i in range(n_train)]
	
	X_train = np.hstack((ones_vector, X_train))
	X_test = np.hstack(([[1] for i in range(n_test)], X_test))

	w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train


	trainErr = errCalc(X_train, y_train, w)
	testErr = errCalc(X_test, y_test, w)

	return trainErr, testErr
	# The training error is: 0.2819998086970962 
	# The testing error is:  0.2803646021141735


def main():

	# import white whine data, X input set, y is the output set
	X, y = cp.load(open('winequality-white.pickle','rb'))

	# splitting up data
	n, d = X.shape

	n_train = int(n * 0.8)
	n_test = n - n_train

	X_train = X[ : n_train]
	y_train = y[ : n_train]

	X_test = X[n_train : ]
	y_test = y[n_train : ]

	



