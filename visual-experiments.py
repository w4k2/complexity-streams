"""
Verified method
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import medfilt
from scipy.ndimage import convolve1d
from tqdm import tqdm
from scipy import stats
from sklearn.svm import OneClassSVM
from sklearn.base import clone
from config import *
np.set_printoptions(
    precision=3,
    suppress=True
)
np.random.seed(1410)

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)

"""
Configure processing parameters
"""
n_classifiers = 20
treshold = 1
bagging_factor = .5
base_clf = OneClassSVM(kernel='rbf')

# Prepare plot
bw = 9
fig, ax = plt.subplots(len(drift_types), 2, figsize=(bw, bw*1.618), sharex=True)

# Select the solution file
dimensionality = dimensionalities[2]
clusters = number_of_clusters[2]
for replication in range(10):
    # Storage for transformed metrics
    for drift_idx, drift_type in enumerate(drift_types):
        supports = []
        filename = '%s_f%i_c%i_r%i' % (
            drift_type,
            dimensionality, clusters, replication
        )
        print('# Processing the %s.' % filename)

        # Load complexities [chunk_id, measure_id], measures [measure_id] and time [chunk_id]
        data = np.load('complexities/%s.npz' % filename)
        complexities, measures, times = [data[k] for k in ['complexities', 'measures','times']]
        metric_filter = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        complexities = complexities[:,metric_filter]
        measures = measures[metric_filter]

        # Gather the basic info
        n_chunks, n_measures = complexities.shape
        print('# %i chunks with %i measures' % (n_chunks, n_measures))
        
        """
        Do the main processing loop
        """
        # Processing parameters
        drifts = []
        ensemble = []
        last_drift = 0
    
        for chunk_id, complexity_vector in enumerate(tqdm(complexities)):
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
                
        print(dimensionality)
        print(clusters)
        print(replication)
        print(drift_idx)
        print('supports', np.array(supports).shape)
        print('drifts', drifts)
        exit()
            
        """
        Presentation
        """
        drfs = find_real_drift(n_chunks, 7)

        ax[drift_idx,0].plot(supports, alpha=.25, c='red')
        ax[drift_idx,0].set_ylim(-treshold*2, treshold*2)
        ax[drift_idx,0].set_yticks([-treshold, 0, treshold])
                
        ax[drift_idx,1].set_xticks(drfs)
        ax[drift_idx,1].set_xlim(0, n_chunks)
        ax[drift_idx,1].vlines(drifts, replication, replication+1, color='red', lw=1)
        #ax[drift_idx,1].scatter(drifts, [replication+.5 
        #                                 for d in drifts], color='red', marker='+')

        if replication == 0:
            ax[drift_idx,0].set_xticks(drfs)
            ax[drift_idx,0].grid(ls=":")

            ax[drift_idx,1].vlines(drfs, -1, 0, color='black', lw=2)
            ax[drift_idx,1].grid(ls=":")
            ax[drift_idx,1].set_ylim(-1, 10)            
            ax[drift_idx,1].set_yticks(np.linspace(-.5,9.5,11),
                                       ['real']+['r%i' % (i+1) for i in range(10)])
        
            for aa in ax.ravel():
                aa.spines['top'].set_visible(False)
                aa.spines['bottom'].set_visible(False)
                aa.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('foo.png')