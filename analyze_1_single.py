from config import *
import numpy as np
import matplotlib.pyplot as plt
from methods import find_real_drift
from strlearn.streams import StreamGenerator

# Gather stream dynamics info
sc = { 'n_features':1, 'n_informative':1,'n_clusters_per_class':1 }

# print(list(drift_types.keys())[0])
# exit()
cp = []
for drift_type in drift_types:
    print(drift_type)
    s = StreamGenerator(**static, **drift_types[drift_type], **sc)
    s._make_classification()
    cp.append(s.concept_probabilities)
    break

cps = np.linspace(0, static['n_chunks'], 
                  static['n_chunks']*static['chunk_size'])


# Establish base stream info
n_chunks = static['n_chunks']
chunk_size = static['chunk_size']
n_clusters_per_class = number_of_clusters[0]
detectors = ['DDM', 'EDDM', 'ADWIN', 'HDDM_A', 'HDDM_W', 'C2D']
drifts = find_real_drift(2000, 7)

# Load results
detection_results = np.load('results/exp_comparison_all.npy')
print(detection_results.shape)
print(np.sum(detection_results[:,:,0,1], axis=-1))
#(replications, dimensionalities, drift_types, n_detectors, n_chunks-1)

fig, aa = plt.subplots(1, 1, figsize=(7,4))

r = detection_results[:,0,0]
r[r==1]=0
print(r.shape)
for d in range(len(detectors)):
    for rep in range(10):
        step = 1
        start = d*10 + step*rep
        stop = d*10 + step*(rep+1)
        detections = np.argwhere(r[rep,d]==2).flatten()
        aa.vlines(detections, start, stop, color='black' if d < 5 else 'red')
        
aa.plot(cps, cp[0]*5-7.5, c='red')
aa.grid(ls=":")

aa.set_title('%i features' % (dimensionalities[0]))
aa.set_ylabel('%s' % list(drift_types.keys())[0], fontsize=12)
aa.set_xticks(drifts, ['D%i' % i for i in range(7)])
aa.set_yticks([(10*i)-5 
                for i in range(len(detectors)+1)], 
                ['concept']+detectors)
aa.spines['top'].set_visible(False)
aa.spines['right'].set_visible(False)
aa.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/e1_single.png')
plt.savefig('figures/e1_single.eps')
plt.savefig('foo.png')