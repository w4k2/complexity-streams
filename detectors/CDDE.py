import numpy as np
from sklearn import clone
from sklearn.svm import OneClassSVM

class CDDE:
    def __init__(self, measures, n_classifiers=1, treshold=1.5, bagging_factor=0.75, base_clf=OneClassSVM(kernel='rbf')):
        self.measures = measures
        self.n_classifiers = n_classifiers
        self.treshold = treshold
        self.bagging_factor = bagging_factor
        self.base_clf = base_clf

        self.drift = []
        self.ensemble = []
        self.supports =[]
        self.last_drift = 0
        self.complexities=[[] for _ in range(len(self.measures))]
        
    
    def feed(self, X, y, pp=None):
        start = self.last_drift
        _X = (np.array(self.complexities)[:,start:]).T

        # Verify if there are any patterns accumulated
        if _X.shape[0] > 0:
            if len(self.ensemble) == 0:
                # If ensemble is empty, build new OC Classifier on FULL DATA
                clf = clone(self.base_clf).fit(_X)
            else:
                # Otherwise, extend the ensemble with a new OC Classifier on resampled data according to bagging factor
                mask = np.random.uniform(size=(_X.shape[0])) < self.bagging_factor    # here bagging
                mask[np.random.randint(_X.shape[0])] = True                      # here ensuring at least one truth

                clf = clone(self.base_clf).fit(_X[mask])
                
            self.ensemble.append(clf)
            
        for m_id, m in enumerate(self.measures):
            v = m(X, y)
            if np.isnan(v):
                v=1
            self.complexities[m_id].append(v)
            
        # Gather and integrate decision
        if len(self.ensemble) > 0:
            support = np.mean([clf.decision_function([np.array(self.complexities)[:,-1]]) for clf in self.ensemble])
        
            if support < -self.treshold:
                # Here is drift
                self.drift.append(2)
                self.last_drift = len(self.complexities[0])-1
                self.ensemble = []
            else:
                self.drift.append(0)
            
            self.supports.append(support)
        else:
            self.supports.append(np.nan)
            self.drift.append(0)
            
        if len(self.ensemble) > self.n_classifiers:
            del self.ensemble[0]