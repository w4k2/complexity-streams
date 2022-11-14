"""
Rysuje, jak zmienia się wartosc wybranych metryk trudności w toku przetwarzania
"""

import numpy as np
import problexity as px
import strlearn as sl
import matplotlib.pyplot as plt
from methods import find_real_drift
from scipy.signal import medfilt

measures = [px.f1, px.f1v, px.l1, px.n1, px.n3, px.n4, px.t1]

stream_static = {
        'n_drifts': 2,
        'n_chunks': 200,
        'chunk_size': 200,
        'n_features': 15,
        'n_informative': 15,
        'n_redundant': 0,
        'recurring': False
    }

drf = find_real_drift(stream_static['n_chunks'],stream_static['n_drifts'])


stream_drfs = {
    'sudden': {},
    'gradual': {
    'concept_sigmoid_spacing': 5
        },
    'incremental': {
    'concept_sigmoid_spacing': 5,
    'incremental': True
        },
    }

m_all = []
for drift_type_id, drift_type in enumerate(stream_drfs):
    config = {
        **stream_static,
        **stream_drfs[drift_type]}
    stream = sl.streams.StreamGenerator(**config)
    
    m_arr =[[] for m in measures]

    for chunk_id in range(stream_static['n_chunks']):
        c = stream.get_chunk()

        for m_id, m in enumerate(measures):
            m_arr[m_id].append(m(c[0], c[1]))
    
    m_all.append(m_arr)

m_all = np.array(m_all)
print(m_all.shape)

fig, ax = plt.subplots(len(measures), 3, figsize=(10,10), dpi=200, sharex=True)

for drift_type_id, drift_type in enumerate(stream_drfs):
    for measure_id, measure in enumerate(measures):
        min_measure = np.min(m_all[:, measure_id])
        max_measure = np.max(m_all[:, measure_id])

        axx = ax[measure_id, drift_type_id]

        axx.plot(m_all[drift_type_id, measure_id], c='black', alpha=0.3, linewidth=1)
        axx.plot(medfilt(m_all[drift_type_id, measure_id],7), c='black', linewidth=1)
        axx.vlines(drf, min_measure, max_measure, color='r', ls=':')
        axx.grid(ls=":")
        
        if drift_type_id==0:
            axx.set_ylabel(measure.__name__)
        if measure_id==0:
            axx.set_title(drift_type)
        if measure_id==len(measures)-1:
            axx.set_xlabel('chunk')
        
        axx.spines['top'].set_visible(False)
        axx.spines['right'].set_visible(False)
        
        axx.set_xlim(0,200)
        axx.set_ylim(min_measure, max_measure)
        ticks=np.round(np.linspace(min_measure, max_measure, 3),2)
        
        if drift_type_id==0:
            axx.set_yticks(ticks)
        else:
            axx.set_yticks(ticks, ['' for v in ticks])


plt.tight_layout()
plt.savefig('figures/complexity_plot.png')
plt.savefig('foo.png')
