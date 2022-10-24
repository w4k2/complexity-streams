"""
Verified method
"""

from cv2 import mean
import numpy as np
import matplotlib.pyplot as plt
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

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)

# Prepare plot
fig, ax = plt.subplots(len(drift_types), 2, figsize=(15,20))

# Select the solution file
dimensionality = dimensionalities[0]
clusters = number_of_clusters[1]
for replication in range(10):
    # Storage for transformed metrics
    supports = []
    for drift_idx, drift_type in enumerate(drift_types):
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
        # Define the processing parameters
        """
        treshold = 1
        base_clf = OneClassSVM(kernel='rbf')
        
        """
        Do the main processing loop
        """
        drifts = []
        n_classifiers = 20
        ensemble = []
        last_drift = 0
        
        for chunk_id, complexity_vector in enumerate(tqdm(complexities)):
            # Gather training space
            start = last_drift
            stop = np.clip(chunk_id-1,0,None)
            
            X = complexities[start:stop]

            # Build model if enough elements
            if len(ensemble) == 0:
                if X.shape[0] > 0:
                    clf = clone(base_clf).fit(X)
                    ensemble.append(clf)
            else:
                bagging_factor = .5
                mask = np.random.uniform(size=(X.shape[0])) < bagging_factor
                mask[np.random.randint(X.shape[0])] = True
                #print(mask)
                clf = clone(base_clf).fit(X[mask])
                ensemble.append(clf)
                
            # Gather and integrate decision
            if len(ensemble) > 0:
                support = np.mean([clf.decision_function([complexity_vector]) 
                                for clf in ensemble])
            
                if support < -treshold:
                    drifts.append(chunk_id)
                    last_drift = chunk_id
                    ensemble = []
                    print('DRIFT ON support = %.3f' % support)
                    #print('!drift')
                
                supports.append(support)
                
            else:
                supports.append(np.nan)
                
            if len(ensemble) > n_classifiers:
                del ensemble[0]
                    
            
        print(drifts)
            
        """
        Presentation
        """
        drfs = find_real_drift(n_chunks, 7)

        ax[drift_idx,1].set_xticks(drfs)
        ax[drift_idx,1].vlines(drfs, -1, 0, color='black')
        ax[drift_idx,1].vlines(drifts, replication, replication+1, color='red')
        ax[drift_idx,1].grid(ls=":")
        ax[drift_idx,1].set_title('Black-real, red-detected %s' % drift_type)
        ax[drift_idx,1].set_xlim(0, n_chunks)

        #tcomp_image = np.copy(XX_t)[:,0,:].T
        ax[drift_idx,0].plot(supports)
        ax[drift_idx,0].set_ylim(-treshold, treshold)


        #comp_image = np.copy(complexities).T
        #comp_image -= np.mean(comp_image, axis=1)[:,None]
        #comp_image /= np.std(comp_image, axis=1)[:,None]

        #ax[drift_idx,2].imshow(comp_image, aspect=n_chunks/1.618/n_measures, interpolation='none', cmap='bwr')
        #ax[drift_idx,2].set_yticks(np.linspace(0,n_measures-1,n_measures))
        #ax[drift_idx,2].set_xticks(drfs)
        #ax[drift_idx,2].grid(ls=":")
        #ax[drift_idx,2].vlines(drfs, 0, len(measures)-1, color='white')

        plt.tight_layout()
        plt.savefig('foo.png')