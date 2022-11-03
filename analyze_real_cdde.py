import numpy as np
import matplotlib.pyplot as plt
import os
import problexity as px

dir = 'real_streams_res/'
for _,_,files in os.walk(dir):
    pass

chunks=200

measures = np.array([getattr(px.classification, n) 
            for n in px.classification.__all__])
metric_mask = np.ones_like(measures).astype(bool)
metric_mask[4] = False
measures = measures[metric_mask]

th = [0.75 for i in range(9)]
th[-1]=1.5

fig, ax = plt.subplots(3,3, figsize=(18,6))
ax=ax.ravel()

c=0
for f in files:
    if f.split('_')[0]!='cdde':
        continue
    
    t=th[c]

    res = np.load('%s/%s' % (dir, f))
    print(res.shape)
    
    ax[c].set_title(f[5:].split('.')[0])
    ax[c].plot(res[:,1], c='tomato')
    d = np.argwhere(res[:,0]==2).flatten()
    ax[c].vlines(d, 0.5, 2*t, color='black')
    ax[c].hlines(-t, 0, len(res[:,1]), color='gray', ls=':')
    ax[c].set_yticks([0,-t], [0,'-t'])
    ax[c].spines['top'].set_visible(False)
    ax[c].spines['right'].set_visible(False)
    
    c+=1
    
ax[-1].set_xlabel('chunk')    
plt.tight_layout()
plt.savefig('real_figures/detections.png')
    