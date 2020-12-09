import numpy as np
from scipy import stats
from collections import Counter
from collections import defaultdict

class NBC:

    feature_types = []
    num_classes = 0
    p = {}
    theta = defaultdict(list)

    def __init__(self, feature_types, num_classes):
        self.feature_types = feature_types
        self.num_classes = num_classes



    def fit(self, X, y):
        # calculate probabilities of p_c
        count = Counter(y)
        (N, ) = y.shape
        self.p = { c : N_c / N for c, N_c in dict(count).items()}

        # calculate theta 
        classInfo =  zip(y, X)
        res = defaultdict(list)

        for y_i, row in classInfo:
            res[y_i].append(row)
        
        for c, rows in res.items():
            rowsM = np.array(rows)

            for col in rowsM.T:
                mean_i = np.average(col)
                sd_i = np.std(col)
                sd_i = max(10 ** -6, sd_i)

                self.theta[c].append((mean_i, sd_i))

        

            
        


    def predict(self, X):
        
        yhat = []
        for x_i in X:
            maxProb = -10000
            maxClass = 0

            for p_c, (c, theta_c) in list(zip(self.p.values(), self.theta.items())):
                prob = p_c
                means, stdevs = tuple(zip(*theta_c))
                prob *= np.prod(stats.norm.pdf(x_i, means, stdevs))
                
                if(maxProb < prob):
                    maxProb = prob
                    maxClass = c

            yhat.append(maxClass) 

        return yhat


def main():
    X = np.array([ [i, i + 1]  for i in range(20)])
    y = np.array([i // 7 for i in range(20)])

    nbc = NBC(['r', 'r'], 3)
    nbc.fit(X, y)

    nbc.predict(np.array([[3.5, 4.5], [10, 3]]))


main()