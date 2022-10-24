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
fig, ax = plt.subplots(len(drift_types), 3, figsize=(15,20))

# Select the solution file
dimensionality = dimensionalities[0]
clusters = number_of_clusters[0]
revision = 2
for drift_idx, drift_type in enumerate(drift_types):
    filename = '%s_f%i_c%i_r%i' % (
        drift_type,
        dimensionality, clusters, revision
    )
    print('# Processing the %s.' % filename)

    # Load complexities [chunk_id, measure_id], measures [measure_id] and time [chunk_id]
    data = np.load('complexities/%s.npz' % filename)
    complexities, measures, times = [data[k] for k in ['complexities', 'measures','times']]

    #metric_filter = [0,1,5,7,10,12,15]
    #complexities = complexities[:,metric_filter]
    #measures = measures[metric_filter]

    # Gather the basic info
    n_chunks, n_measures = complexities.shape
    print('# %i chunks with %i measures' % (n_chunks, n_measures))

    """
    # Define the processing parameters
    """
    horizon = 50
    n_models = 10
    treshold = 5
    ensemble = []
    base_clf = OneClassSVM()
    
    """
    Do the main processing loop
    """
    drifts = []
    last_drift = 0
    for chunk_id, complexity_vector in enumerate(tqdm(complexities)):
        # Gather training space
        start = last_drift
        stop = np.clip(chunk_id-1,0,None)
        
        X = complexities[start:stop]

        # Build model if enough elements
        if X.shape[0] > horizon:
            ensemble.append(clone(base_clf).fit(X))
            
        # Gather and integrate decision
        decision_vector = np.array([clf.decision_function([complexity_vector]) for clf in ensemble])
            
        
        if len(decision_vector) > 0 and np.abs(np.mean(decision_vector)) > treshold:
            drifts.append(chunk_id)
            last_drift = chunk_id
            ensemble = []
            print(decision_vector)
            #print('!drift')
            
        # Prune ensemble
        if len(ensemble) > n_models:
            del ensemble[0]
        
        #if chunk_id > 450:
        #    break
        
    print(drifts)
        
    """
    Presentation
    """
    drfs = find_real_drift(n_chunks, 7)

    ax[drift_idx,1].set_xticks(drfs)
    ax[drift_idx,1].vlines(drfs, .5, 1, color='black')
    ax[drift_idx,1].vlines(drifts, 0, .5, color='red')
    ax[drift_idx,1].grid(ls=":")
    ax[drift_idx,1].set_title('Black-real, red-detected %s' % drift_type)


    comp_image = np.copy(complexities).T
    #comp_image -= np.mean(comp_image, axis=1)[:,None]
    #comp_image /= np.std(comp_image, axis=1)[:,None]

    ax[drift_idx,2].imshow(comp_image, aspect=n_chunks/1.618/n_measures, interpolation='none', cmap='bwr')
    ax[drift_idx,2].set_yticks(np.linspace(0,n_measures-1,n_measures))
    ax[drift_idx,2].set_xticks(drfs)
    #ax[drift_idx,2].grid(ls=":")
    ax[drift_idx,2].vlines(drfs, 0, len(measures)-1)

    plt.tight_layout()
    plt.savefig('foo.png')