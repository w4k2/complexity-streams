import os
import strlearn as sl
import problexity as px
import numpy as np
from detectors.CDDE import CDDE

dir = 'real_streams/'
for _,_,files in os.walk(dir):
    pass

chunks = 2000
chunk_size = 250

measures = np.array([getattr(px.classification, n) 
            for n in px.classification.__all__])
metric_mask = np.ones_like(measures).astype(bool)
metric_mask[4] = False
measures = measures[metric_mask]

for f in files:
    print(f)
    
    if f.split('.')[0] == 'electricity':
        if f.split('.')[1]=='npy':
            continue
        data = np.loadtxt('%s/%s' % (dir, f), delimiter=',',skiprows=1, dtype=object)
        data[data=='UP'] = 1
        data[data=='DOWN'] = 0
        data = data.astype(float)
        np.save('%s/electricity.npy' % dir, data)
        stream = sl.streams.NPYParser('%s/electricity.npy' % dir, chunk_size=chunk_size, n_chunks=chunks)
    else:
        stream = sl.streams.ARFFParser('%s/%s' % (dir, f), chunk_size=chunk_size, n_chunks=chunks)


    cdde = CDDE(measures= measures, thresolh=1)
    for chunk in range(chunks):
        try:
            X, y = stream.get_chunk()
            print(np.unique(y))
        except:
            print(chunk, 'break')
            break
        
        cdde.feed()

    np.save('real_streams_res/cdde_%s' % f.split('.')[0], np.array(c))
    # print(c)