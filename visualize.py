import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import convolve1d
from torch import mean

filename = 'bal_sudden_f8_c2_r0'
print(filename)

data = np.load('complexities/%s.npz' % filename)
print(data)
print(data.files)

complexities, measures = data['complexities'], data['measures']
times = data['times']

print(complexities, complexities.shape, measures)

bw = 15
fig, ax = plt.subplots(2,2,figsize=(bw, bw))

# Prepare convolution weights
window_size = 20
weights = np.ones(window_size)/window_size
print(weights)

for m_idx, (measure, vector) in enumerate(zip(measures, complexities.T)):
    mvector = medfilt(vector, 1)
    ax[0,0].plot(mvector, label=measure)
    
    # Convolve
    mean_value_vector = convolve1d(vector, weights)
    norm_vector = vector - mean_value_vector
    
    print(vector.shape)
    print(mean_value_vector.shape)
    
    ax[0,1].plot(mean_value_vector)
    ax[1,1].plot((norm_vector*2)+m_idx, c='black')

ax[1,0].plot(medfilt(times,1))    
ax[0,0].legend()
ax[1,1].set_yticks(np.linspace(0,21,22), measures)

# Ranger
for aa in ax.ravel():
    aa.set_xlim(0,1000)

plt.tight_layout()
plt.savefig('foo.png')