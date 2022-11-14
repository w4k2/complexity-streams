import numpy as np
from sklearn.base import clone

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)

def dderror(drifts, detections, n_chunks):

    if len(detections) == 0: # no detections
        detections = np.arange(n_chunks)

    n_detections = len(detections)
    n_drifts = len(drifts)

    ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    d1metric = np.mean(cdri)
    d2metric = np.mean(cdec)
    cmetric = np.abs((n_drifts/n_detections)-1)

    return d1metric, d2metric, cmetric
    # d1 - detection from nearest drift
    # d2 - drift from nearest detection

def process(complexities,
            n_classifiers,
            base_clf,
            treshold,
            bagging_factor):
    # Processing parameters
    supports = []
    drifts = []
    ensemble = []
    last_drift = 0

    for chunk_id, complexity_vector in enumerate(complexities):
        # Establish training interval and gather training set
        start = last_drift
        stop = np.clip(chunk_id-1,0,None)
        X = complexities[start:stop]

        # Verify if there are any patterns accumulated
        if X.shape[0] > 0:
            if len(ensemble) == 0:
                # If ensemble is empty, build new OC Classifier on FULL DATA
                clf = clone(base_clf).fit(X)
            else:
                # Otherwise, extend the ensemble with a new OC Classifier on resampled data according to bagging factor
                mask = np.random.uniform(size=(X.shape[0])) < bagging_factor    # here bagging
                mask[np.random.randint(X.shape[0])] = True                      # here ensuring at least one truth

                clf = clone(base_clf).fit(X[mask])
                
            ensemble.append(clf)
            
        # Gather and integrate decision
        if len(ensemble) > 0:
            support = np.mean([clf.decision_function([complexity_vector]) 
                            for clf in ensemble])
        
            if support < -treshold:
                # Here is drift
                drifts.append(chunk_id)
                last_drift = chunk_id
                ensemble = []
            
            supports.append(support)
        else:
            supports.append(np.nan)
            
        if len(ensemble) > n_classifiers:
            del ensemble[0]

    # Postprocess
    supports = np.array(supports)
    drifts_vec = np.zeros_like(supports)
    drifts_vec[drifts] = 2
            
    return supports, drifts, drifts_vec

dets = np.array([20, 50, 100])
drfs = np.array([10, 50, 90, 100])
print(dderror(drfs,dets,120))