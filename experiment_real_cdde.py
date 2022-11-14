import os
import strlearn as sl
import problexity as px
import numpy as np
from detectors.CDDE import CDDE

dir = 'real_streams/'
files = [
    'covtypeNorm-1-2vsAll-pruned.arff',
    'electricity.csv',
    'poker-lsn-1-2vsAll-pruned.arff',
    'INSECTS-abrupt_imbalanced_norm.arff',
    'INSECTS-gradual_imbalanced_norm.arff',
    'INSECTS-incremental_imbalanced_norm.arff'
]

chunks = 2000
chunk_size = 250

measures = np.array([getattr(px.classification, n) 
            for n in px.classification.__all__])
metric_mask = np.ones_like(measures).astype(bool)
metric_mask[4] = False
measures = measures[metric_mask]

th = 0.75

c=0
for f in files:
    print(f)
    
    if f.split('.')[0] == 'electricity':
        data = np.loadtxt('%s/%s' % (dir, f), delimiter=',',skiprows=1, dtype=object)
        data[data=='UP'] = 1
        data[data=='DOWN'] = 0
        data = data.astype(float)
        np.save('%s/electricity.npy' % dir, data)
        stream = sl.streams.NPYParser('%s/electricity.npy' % dir, chunk_size=chunk_size, n_chunks=chunks)
    else:
        stream = sl.streams.ARFFParser('%s/%s' % (dir, f), chunk_size=chunk_size, n_chunks=chunks)

    if f.split('-')[0]=='poker':
        cdde = CDDE(measures=measures, treshold=1.5)
    else:
        cdde = CDDE(measures=measures, treshold=th)
        
    for chunk in range(chunks):
        try:
            X, y = stream.get_chunk()
        except:
            print(chunk, 'break')
            break
        
        if len(np.unique(y))!=2:
            print('continue')
            continue
        
        cdde.feed(X, y)
        
    r = np.concatenate((np.array(cdde.drift).reshape(-1, 1), np.array(cdde.supports).reshape(-1, 1)), axis=1)

    print(r[:100])
    np.save('real_streams_res/cdde_%s' % f.split('.')[0], r)
    c+=1