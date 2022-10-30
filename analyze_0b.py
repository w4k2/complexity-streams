from cv2 import threshold
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
_n_classifiers = [1]
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
for nt_idx, treshold in enumerate(_treshold):
    for nb_idx, bagging_factor in enumerate(_bagging_factor):
        # Establish config filename
        config_filename = 'e%i_t%i_b%i' % (1, int(treshold*100), int(bagging_factor*100))

        # Prepare results cube [DIMENSIONALITY x REPLICATION x DRIFT TYPE x 2[SUPPORT, DRIFT]] 
        nnn = 2000 # To remove and replace with 2000
        results = np.load('results/%s.npy' % config_filename).squeeze()

        #print('# On config\n  %s' % config_filename)
        
        fig, ax = plt.subplots(len(drift_types), len(dimensionalities), figsize=(10,10), sharex=True, sharey=True)
        
        for dimensionality_idx, dimensionality in enumerate(dimensionalities):
            for replication in range(10):
                for drift_idx, drift_type in enumerate(drift_types):
                    support, detections = results[dimensionality_idx, replication, drift_idx]
                    detections = np.where(detections)[0]
                    
                    aa = ax[drift_idx,dimensionality_idx]
                    ax[drift_idx,dimensionality_idx].plot(support, c='red', alpha=.1)
                    ax[drift_idx,dimensionality_idx].set_ylim(-treshold*3, treshold*3)
                    ax[drift_idx,dimensionality_idx].set_xticks(drifts, ['D%i' % i for i in range(7)])
                    
                    step = treshold/5
                    start = treshold + step*replication
                    stop = treshold + step*(replication+1)
                    aa.vlines(detections, start, stop, color='black')
                    #print(detections)
                    
                    aa.set_yticks([-treshold, 0, treshold], ['t', '0', 't'])
                    aa.grid(ls=":")
                    aa.spines['top'].set_visible(False)
                    aa.spines['right'].set_visible(False)
                    aa.set_title('%s | %id' % (drift_type, dimensionality))
                 
        plt.suptitle('t=%.3f, bf=%.3f' % (treshold, bagging_factor))
        plt.tight_layout()
        filename = 'figures/e0_runs_%i_%i.png' % (nt_idx, nb_idx)   
        print(filename)
        plt.savefig('foo.png')   
        plt.savefig(filename)
