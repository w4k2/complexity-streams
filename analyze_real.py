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

for f in files:
    if f.split('_')[0]=='cdde':
        continue    
    res = np.load('%s/%s' % (dir, f))
    res_det = np.load('%s/cdde_%s' % (dir, f))
    print(np.sum(np.isnan(res)), f, res.shape)
    
    fig, ax = plt.subplots(3, 7, figsize=(15,6), sharex=True, sharey=True)
    ax = ax.ravel()
    plt.suptitle(f.split('.')[0], fontsize=15)
    for i in range(21):
        ax[i].plot(res[i], c='cornflowerblue')
        ax[i].vlines(np.argwhere(res_det[:,0]==2).flatten(), 0, 1, color='tomato', ls=":", alpha=0.75)
        ax[i].set_title(measures[i].__name__)
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('real_figures/%s.png' % f.split('.')[0])
    # exit()