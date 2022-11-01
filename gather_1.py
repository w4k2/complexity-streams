import numpy as np
from config import *

treshold=1.46
bagging_factor=0.75
config_filename = 'e%i_t%i_b%i' % (1, int(treshold*100), int(bagging_factor*100))

# dimensionalities x replications x drift_types x n_chunks-1
drifts_cdde = np.load('results/%s.npy' % config_filename)[0,:,:,:,1,:-1]
drifts_cdde = drifts_cdde.swapaxes(0,1)
print(drifts_cdde.shape)
print(drifts_cdde[0,0,0,:500])

# replications, dimensionalities, drift_types, n_detectors, n_chunks-1
drifts_others = np.load('results/exp_comparison.npy')
print(drifts_others.shape)

drifts_all = np.concatenate((drifts_others, drifts_cdde[:,:,:,np.newaxis]), axis=3)
print(drifts_all.shape)

np.save('results/exp_comparison_all', drifts_all)