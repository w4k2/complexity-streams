from config import *
import numpy as np
import matplotlib.pyplot as plt
from methods import find_real_drift
from strlearn.streams import StreamGenerator

# Gather stream dynamics info
sc = { 'n_features':1, 'n_informative':1,'n_clusters_per_class':1 }

cp = []
for drift_type in drift_types:
    print(drift_type)
    s = StreamGenerator(**static, **drift_types[drift_type], **sc)
    s._make_classification()
    cp.append(s.concept_probabilities)

cps = np.linspace(0, static['n_chunks'], 
                  static['n_chunks']*static['chunk_size'])


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
                aa.vlines(detections, start, stop, color='black' if d < 4 else 'red')
                
        aa.plot(cps, cp[dt_id]*5-7.5, c='red')
        aa.grid(ls=":")
                    
        aa.set_title('%i dim | %s' % (dim, dt))
        aa.set_xticks(drifts, ['D%i' % i for i in range(7)])
        aa.set_yticks([(10*i)-5 
                       for i in range(len(detectors)+1)], 
                      ['concept']+detectors)
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
        aa.spines['bottom'].set_visible(False)
        
    plt.tight_layout()
    plt.savefig('figures/e1.png')
    plt.savefig('figures/e1.eps')
    plt.savefig('foo.png')