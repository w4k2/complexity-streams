import numpy as np
from config import *
import matplotlib.pyplot as plt

# Main configuration
tested_params = ['#classifiers', 'treshold', 'bagging_factor']
tested_ranges = [
    np.linspace(1, 20, 5).astype(int),
    np.around(np.linspace(0.2, 4, 10), 2),
    np.linspace(0.25, 0.75, 3)
]
n_tested_params = len(tested_params)

# Overview params
overview_params = ['drift_types', 'dimensionalities']
overview_iterators = [drift_types, dimensionalities]

# Load scores and flatten it by replications
scores = np.load('scores/e0.npy')
mean_scores = np.mean(scores, axis=4)

val_formats = ['%.0f', '%.0f', '%.1f']
vals = ['d1', 'd2', 'cm']

# Do the comparisons
for a, b in zip([0,0,2], [1,2,1]):
    pool = [a, b]
    
    param_a = tested_params[a]
    param_b = tested_params[b]
    
    range_a = tested_ranges[a]
    range_b = tested_ranges[b]
    
    print('\n# %s vs %s' % (param_a, param_b), a, b)
    print(param_a, range_a)
    print(param_b, range_b)
    
    # Verify which to flatten
    to_flatten = []
    for flat in range(n_tested_params):
        if flat not in pool:
            to_flatten.append(flat)
    
    proto_scores = np.mean(mean_scores, axis=tuple(to_flatten))
    
    print(proto_scores.shape)
    
    # Prepare plot
    for v_idx, (v_name, v_format) in enumerate(zip(vals, val_formats)):
        # Calculate midpoint
        minpoint = np.min(proto_scores[:,:,:,:,v_idx])
        maxpoint = np.max(proto_scores[:,:,:,:,v_idx])
        midpoint = np.mean(proto_scores[:,:,:,:,v_idx])
        
        fig, ax = plt.subplots(*[len(it) for it in overview_iterators], figsize=(12,12),
                            sharex=True, sharey=True)
        for op_idx_a, op_a in enumerate(overview_iterators[0]):
            for op_idx_b, op_b in enumerate(overview_iterators[1]):
                print(op_a, op_b)
                aa = ax[op_idx_a, op_idx_b]
                aa.set_title('%i dim | %s' % (op_b, op_a))
                
                image = proto_scores[:,:,op_idx_b,op_idx_a,v_idx]
                if a > b:
                    image = image.T
                print(image.shape)
                
                aa.imshow(image, cmap='bone', vmin=minpoint, vmax=maxpoint)
                
                for j, row in enumerate(image):
                    for k, v in enumerate(row):
                        aa.text(k,j,v_format % v, ha='center', va='center',
                                c='white' if v < midpoint else 'black')
                
                aa.set_yticks(list(range(len(tested_ranges[a]))), tested_ranges[a])
                aa.set_xticks(list(range(len(tested_ranges[b]))), ['%.1f' % v for v in tested_ranges[b]])
        
        fig.suptitle('%s vs %s on %s' % (param_a, param_b, v_name))
        plt.tight_layout()
        plt.savefig('figures/e0_cmp_%i_%i_m%i.png' % (a, b, v_idx))
        plt.savefig('foo.png')

        
    
    