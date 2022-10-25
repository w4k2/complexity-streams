"""
Verified method
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import medfilt
from scipy.ndimage import convolve1d
from sklearn import config_context
from scipy import stats
from sklearn.svm import OneClassSVM
from config import *
from methods import process
from tqdm import tqdm
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
_n_classifiers = np.linspace(1, 30, 10).astype(int)
_treshold = np.linspace(0.2, 5, 10)
_bagging_factor = np.linspace(0.1, 0.9, 5)

base_clf = OneClassSVM(kernel='rbf')
for n_classifiers in _n_classifiers:
    for treshold in _treshold:
        for bagging_factor in _bagging_factor:
            # Establish config filename
            config_filename = 'e%i_t%i_b%i' % (n_classifiers, int(treshold*100), int(bagging_factor*100))
            print(config_filename)

            # Prepare results cube [CLUSTERS x DIMENSIONALITY x REPLICATION x DRIFT TYPE x 2[SUPPORT, DRIFT]] 
            nnn = 2000 # To remove and replace with 2000
            results = np.zeros((len(number_of_clusters),
                                len(dimensionalities),
                                10,
                                len(drift_types),
                                2,
                                nnn))

            # Select the solution file
            #print(np.product(results.shape)/nnn, results.shape)
            pbar = tqdm(total=np.product(results.shape)/nnn)
            for clusters_idx, clusters in enumerate(number_of_clusters):
                for dimensionality_idx, dimensionality in enumerate(dimensionalities):
                    for replication in range(10):
                        # Storage for transformed metrics
                        for drift_idx, drift_type in enumerate(drift_types):
                            # Get source filename
                            filename = '%s_f%i_c%i_r%i' % (
                                drift_type,
                                dimensionality, 
                                clusters, 
                                replication
                            )

                            # Load complexities [chunk_id, measure_id], measures [measure_id] and time [chunk_id]
                            data = np.load('complexities/%s.npz' % filename)
                            complexities, measures = [data[k] for k in ['complexities', 'measures']]
                            
                            # Filter complexities
                            metric_filter = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
                            complexities = complexities[:,metric_filter]
                            measures = measures[metric_filter]
                            
                            # To remove
                            complexities = complexities[:nnn]

                            # Gather the basic info
                            n_chunks, n_measures = complexities.shape
                            
                            """
                            Do the main processing loop
                            """
                            supports, drifts, drifts_vec = process(complexities,
                                                                n_classifiers,
                                                                base_clf,
                                                                treshold,
                                                                bagging_factor)
                                            
                            # Store results [CLUSTERS x DIMENSIONALITY x REPLICATION x DRIFT TYPE x 2[SUPPORT, DRIFT]] 
                            results[clusters_idx, dimensionality_idx, replication, drift_idx, 0] = supports
                            results[clusters_idx, dimensionality_idx, replication, drift_idx, 1] = drifts_vec
                            
                            pbar.update(1)

            print(results, results.shape)
            np.save('results/%s' % config_filename, results)
