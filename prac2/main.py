import numpy as np
from scipy import stats
from collections import Counter
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pprint


pp = pprint.PrettyPrinter(indent=4)

class NBC:

    feature_types = []
    num_classes = 0
    p = {}
    theta = defaultdict(list)

    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes


    def fit(self, X, y):
        self.theta = defaultdict(list)
        self.p = {}

        # calculate probabilities of p_c
        count = Counter(y)
        (N, ) = y.shape
        self.p = { c : N_c / N for c, N_c in dict(count).items()}

        # calculate theta 
        classInfo =  zip(y, X)
        data_by_class = defaultdict(list)

        # data by class[c] holds all inputs with class c 
        for y_i, row in classInfo:
            data_by_class[y_i].append(row)
        
        for c, data in data_by_class.items():

            data = np.array(data)
            _, d = data.shape
            assert(d == len(self.feature_types))

            for k in range(d):
                col = data[ : , k]

                mean_i = np.average(col)
                sd_i = np.std(col)
                sd_i = max(10 ** -10, sd_i)

                self.theta[c].append((mean_i, sd_i))  


    def predict(self, X):
        
        yhat = []
        for x_i in X:
            maxProb = -1000000
            maxClass = 0

            for p_c, (c, theta_c) in list(zip(self.p.values(), self.theta.items())):
                log_prob = np.log(p_c)
                means, stdevs = tuple(zip(*theta_c))
                means = np.array(means)
                stdevs = np.array(stdevs)

                raw_prob = np.array([max(10 ** -30, x) for x in stats.norm.pdf(x_i, means, stdevs)])
                log_prob += np.sum(np.log(raw_prob))
                
                if(maxProb < log_prob):
                    maxProb = log_prob
                    maxClass = c

            yhat.append(maxClass) 

        return yhat

# returns error
def testModel( model, Xtrain, ytrain, Xtest, ytest):
    model.fit(Xtrain, ytrain)

    yhat = model.predict(Xtest)
    return 1 - np.mean(yhat == ytest)

def run(X, y):

    N, _ = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N)
    
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]


    nbc = NBC(['r', 'r', 'r', 'r'], 3)
    log_reg = LogisticRegression(C = 10, max_iter = 300)

    idxs = [int(Ntrain * k / 10) for k in range(1 , 11)]
    Xtrains = [Xtrain[ : idxs[k], : ] for k in range(10)]
    ytrains = [ytrain[ : idxs[k]] for k in range(10)]
    
    
    NBC_acc_list = [testModel(nbc, Xtrains[k], ytrains[k], Xtest, ytest) for k in range(10)]
    LR_acc_list = [testModel(log_reg, Xtrains[k], ytrains[k], Xtest, ytest) for k in range(10)]

    return NBC_acc_list, LR_acc_list 

def mkLineGraph(NBC_err, LR_err):

	x = range(10, 110, 10)

	plt.plot(x, NBC_err, color='skyblue', linewidth=1.5, label = "Naive Bayes Classifier Error")
	plt.plot(x, LR_err, color='red', linewidth=1.5, label = "Logistic Regression Error")
	plt.xlabel('Percentage of training data')
	plt.ylabel('Error')
	plt.title('NBC vs LR error for different percentages of training data')
	plt.legend()

	plt.show()

def main():

#   Handin 1: As C is the inverse of regularisation strength I would choose C = 10 for lambda = 0.1
    _ = LogisticRegression(C = 10, max_iter = 300)

#   Handin 2: Iris Dataset
    iris = load_iris()
    X, y = iris['data'], iris['target']

#   type of trials is: list of pairs of lists
#   want to get pair of lists of lists
    trials = [run(X,y) for _ in range(1000)]

    NBC_trials = np.array([r[0] for r in trials])
    LR_trials = np.array([r[1] for r in trials])

    NBC_err = [np.average(col) for col in NBC_trials.T]
    LR_err = [np.average(col) for col in LR_trials.T]

    mkLineGraph(NBC_err, LR_err)
    



main()