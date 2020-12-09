import numpy as np
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
        """
        Predict the outputs for input X
        """
        y = []
        return y


def main():
    X = np.array([ [i, i + 1]  for i in range(20)])
    y = np.array([i % 3 for i in range(20)])

    nbc = NBC(['r', 'r'], 3)
    nbc.fit(X, y)

main()