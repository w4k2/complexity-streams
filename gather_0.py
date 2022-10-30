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
from methods import process, dderror, find_real_drift
from tqdm import tqdm
np.set_printoptions(
    precision=3,
    suppress=True
)
np.random.seed(1410)

n_chunks = 2000
n_chunks = 8

drifts = find_real_drift(2000, 7)

"""
Configure processing parameters
"""
_n_classifiers = np.linspace(1, 20, 5).astype(int)
_treshold = np.linspace(0.2, 4, 10)
_bagging_factor = np.linspace(0.25, 0.75, 3)


"""
Review scores
"""
scores = np.zeros((
    len(_n_classifiers),    # 0
    len(_treshold),         # 1
    len(_bagging_factor),   # 2
    len(dimensionalities),  # 3
    10,                     # 4
    len(drift_types),        # 5
    3,                      # d1, d2, cm
))
for nc_idx, n_classifiers in enumerate(_n_classifiers):
    for nt_idx, treshold in enumerate(_treshold):
        for nb_idx, bagging_factor in enumerate(_bagging_factor):
            # Establish config filename
            config_filename = 'e%i_t%i_b%i' % (n_classifiers, int(treshold*100), int(bagging_factor*100))

            # Prepare results cube [DIMENSIONALITY x REPLICATION x DRIFT TYPE x 2[SUPPORT, DRIFT]] 
            nnn = 2000 # To remove and replace with 2000
            results = np.load('results/%s.npy' % config_filename).squeeze()

            #print('# On config\n  %s' % config_filename)
            
            for dimensionality_idx, dimensionality in enumerate(dimensionalities):
                for replication in range(10):
                    for drift_idx, drift_type in enumerate(drift_types):
                        support, detections = results[dimensionality_idx, replication, drift_idx]
                        
                        detections = np.where(detections)[0]
                        
                        d1, d2, cm = dderror(drifts, detections, n_chunks)
                        
                        scores[nc_idx, nt_idx, nb_idx, 
                                   dimensionality_idx, replication, drift_idx] = [d1, d2, cm]
                        
                        #print('- %20s - %.3f %.3f %.3f' % (drift_type, d1, d2, cm))
              
np.save('scores/e0.npy', scores)    
print(scores, scores.shape)  

