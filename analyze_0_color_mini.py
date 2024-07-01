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

print(mean_scores.shape) # classifiers, threshold, bagging, features, drifts, measures

# Prepare plot
fig, ax = plt.subplots(2,2, figsize=(10,4), sharex=True, sharey=True)
ax = ax.ravel()

mean_scores = mean_scores.swapaxes(1,2)

for m_id in range(3):
    image = mean_scores[0,:,:,0,0,m_id]

    ax[m_id].imshow(image, cmap='bone')
    if m_id==0:
        for j, row in enumerate(image):
            for k, v in enumerate(row):
                ax[m_id].text(k,j,'%.1f' % np.mean(v), 
                            ha='center', va='center',
                            c='white' if np.mean(v) < 40 else 'black')
    
    if m_id==1:
        for j, row in enumerate(image):
            for k, v in enumerate(row):
                ax[m_id].text(k,j,'%.1f' % np.mean(v), 
                            ha='center', va='center',
                            c='white' if np.mean(v) < 200 else 'black')
    
    if m_id==2:
        for j, row in enumerate(image):
            for k, v in enumerate(row):
                ax[m_id].text(k,j,'%.1f' % np.mean(v), 
                            ha='center', va='center',
                            c='white' if np.mean(v) < 2 else 'black')
                         
                    
    ax[m_id].set_yticks(list(range(len(tested_ranges[2]))), tested_ranges[2])
    ax[m_id].set_xticks(list(range(len(tested_ranges[1]))), ['%.1f' % v for v in tested_ranges[1]])
    
    if m_id in [0,2]:
        ax[m_id].set_ylabel(tested_params[2])
    if m_id==2:
        ax[m_id].set_xlabel(tested_params[1])


image = mean_scores[0,:,:,0,0]
for d in range(3):
    image[:,:,d]-=np.min(image[:,:,d])
    image[:,:,d]/=np.max(image[:,:,d])
print(image.shape)

ax[-1].imshow(image)

for j, row in enumerate(image):
    for k, v in enumerate(row):
        ax[-1].text(k,j,'%.2f' % np.mean(v), 
                    ha='center', va='center',
                    c='white' if np.mean(v) < 0.5 else 'black')
                
ax[-1].set_yticks(list(range(len(tested_ranges[2]))), tested_ranges[2])
ax[-1].set_xticks(list(range(len(tested_ranges[1]))), ['%.1f' % v for v in tested_ranges[1]])


# ax[-1].set_ylabel(tested_params[2])
ax[-1].set_xlabel(tested_params[1])

ax[0].set_title('Detection from nearest drift (D1)')
ax[1].set_title('Drift from nearest detection (D2)')
ax[2].set_title('Ratio of drifts to detections (R)')
ax[3].set_title('Combined and normalized criteria')

# fig.suptitle('%s vs %s' % (param_a, param_b))
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/c2d_e0_mini.png')
plt.savefig('figures/c2d_e0_mini.eps')