import numpy as np
from methods import dderror

dets = np.arange(10,50) #(5.0, 0.3333333333333333, 0.925)
dets = np.array([10]) #(0.0, 20.0, 2.0)
dets = np.array([]) #(5.1, 0.3333333333333333, 0.94)
drits = np.array([10,30,50])

print(dderror(drits, dets, 50))

