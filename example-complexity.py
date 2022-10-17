import numpy as np

filename = 'bal_sudden_f8_c2_r0'
print(filename)

data = np.load('complexities/%s.npz' % filename)
print(data)
print(data.files)

complexities, measures = data['complexities'], data['measures']

print(complexities, complexities.shape, measures)