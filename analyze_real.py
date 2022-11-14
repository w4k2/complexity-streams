import numpy as np
import matplotlib.pyplot as plt
import os
import problexity as px

dir = 'real_streams_res'
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

for i, f in enumerate(files):
    if f.split('_')[0]=='cdde':
        continue    
    res = np.load('%s/%s.npy' % (dir, f.split('.')[0]))
    res_det = np.load('%s/cdde_%s.npy' % (dir, f.split('.')[0]))
    print(np.sum(np.isnan(res)), f, res.shape)
    
    fig, ax = plt.subplots(3, 7, figsize=(15,6), sharex=True, sharey=True)
    ax = ax.ravel()
    plt.suptitle(files_labels[i], fontsize=15)
    for i in range(21):
        ax[i].plot(res[i], c='black', alpha=0.7, linewidth=1)
        ax[i].vlines(np.argwhere(res_det[:,0]==2).flatten(), 0, 1, color='tomato', ls=":", alpha=0.75)
        ax[i].set_title(measures[i].__name__)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].grid(ls=":", axis='y')
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('real_figures/%s.png' % f.split('.')[0])
    # exit()