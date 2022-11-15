import numpy as np
import matplotlib.pyplot as plt
import os
import problexity as px

dir = 'real_streams_res/'
files = [
    'covtypeNorm-1-2vsAll-pruned.arff',
    'electricity.csv',
    'poker-lsn-1-2vsAll-pruned.arff',
    'INSECTS-abrupt_imbalanced_norm.arff',
    'INSECTS-gradual_imbalanced_norm.arff',
    'INSECTS-incremental_imbalanced_norm.arff'
]

files_labels = [
    'covtype-1-2vsAll',
    'electricity',
    'poker-lsn-1-2vsAll',
    'INSECTS-abrupt',
    'INSECTS-gradual',
    'INSECTS-incremental'
]

chunks=200

measures = np.array([getattr(px.classification, n) 
            for n in px.classification.__all__])
metric_mask = np.ones_like(measures).astype(bool)
metric_mask[4] = False
measures = measures[metric_mask]

th = [0.75 for i in range(9)]
th[2]=1.5

fig, ax = plt.subplots(3,2, figsize=(16,6))
ax=ax.ravel()

c=0
for i, f in enumerate(files):
   
    t=th[c]

    res = np.load('%s/cdde_%s.npy' % (dir, f.split('.')[0]))
    print(res.shape)
    
    ax[c].set_title(files_labels[i])
    ax[c].plot(res[:,1], c='tomato')
    d = np.argwhere(res[:,0]==2).flatten()
    ax[c].vlines(d, 0.5, 2*t, color='black')
    ax[c].hlines(-t, 0, len(res[:,1]), color='gray', ls=':')
    ax[c].set_yticks([0,-t])
    ax[c].spines['top'].set_visible(False)
    ax[c].spines['right'].set_visible(False)
    
    c+=1
    
ax[-1].set_xlabel('chunk')    
ax[-2].set_xlabel('chunk')    
ax[2].set_ylabel('decision function/ detections')    
# ax[2].set_ylabel('decision function')    
# ax[4].set_ylabel('decision function')    
plt.tight_layout()
plt.savefig('real_figures/detections.png')
plt.savefig('real_figures/detections.eps')
plt.savefig('foo.png')
    