from config import *
import numpy as np
import matplotlib.pyplot as plt
from methods import find_real_drift

# Establish base stream info
n_chunks = static['n_chunks']
chunk_size = static['chunk_size']
n_clusters_per_class = number_of_clusters[0]
detectors = ['DDM', 'ADWIN', 'HDDM_A', 'HDDM_W', 'CDDE']
drifts = find_real_drift(2000, 7)

# Load results
detection_results = np.load('results/exp_comparison_all.npy')
#(replications, dimensionalities, drift_types, n_detectors, n_chunks-1)

fig, ax = plt.subplots(len(drift_types), len(dimensionalities), figsize=(12,12), sharex=True, sharey=True)

for dt_id, dt in enumerate(drift_types):
    for dim_id, dim in enumerate(dimensionalities):
        aa = ax[dt_id, dim_id]
        r = detection_results[:,dim_id,dt_id]
        r[r==1]=0
        print(r.shape)
        for d in range(len(detectors)):
            for rep in range(10):
                step = 1
                start = d*10 + step*rep
                stop = d*10 + step*(rep+1)
                detections = np.argwhere(r[rep,d]==2).flatten()
                aa.vlines(detections, start, stop, color='black')
                    
        aa.set_title('%i dim | %s' % (dim, dt))
        aa.set_xticks(drifts, ['D%i' % i for i in range(7)])
        aa.set_yticks([(10*i)+5 for i in range(len(detectors))], detectors)
        
plt.tight_layout()
plt.savefig('figures/e1.png')
plt.savefig('foo.png')