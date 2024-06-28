import os
import strlearn as sl
import problexity as px
import numpy as np


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

for f in files:
    print(f)
    c = [[] for i in range(len(measures))]

    if f.split('.')[0] == 'electricity':
        data = np.loadtxt('%s/%s' % (dir, f), delimiter=',',skiprows=1, dtype=object)
        data[data=='UP'] = 1
        data[data=='DOWN'] = 0
        data = data.astype(float)
        np.save('%s/electricity.npy' % dir, data)
        stream = sl.streams.NPYParser('%s/electricity.npy' % dir, chunk_size=chunk_size, n_chunks=chunks)
    else:
        stream = sl.streams.ARFFParser('%s/%s' % (dir, f), chunk_size=chunk_size, n_chunks=chunks)

    irs = []
    for chunk in range(chunks):
        try:
            X, y = stream.get_chunk()
            # print(X.shape)
            # break
        except:
            # print(chunk, 'break')
            break
        
        if len(np.unique(y))!=2:
            # print('continue')
            continue
        
        c0, c1 = np.unique(y, return_counts=True)[1]
        # print(c0,c1)
        ir = c1/len(y)
        irs.append(ir)
        
    # np.save('real_streams_res/%s' % f.split('.')[0], np.array(c))
    print(f, chunk, X.shape, np.min(irs), np.max(irs))