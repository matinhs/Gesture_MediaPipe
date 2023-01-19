import numpy as np

def label(x):
    x[x>=0.5] = 1
    x[x<0.5] = 0
    if np.array_equiv(x,[0,0,0,1]):
        return 'Class 1'
    if np.array_equiv(x,[0,0,1,0]):
        return 'Class 2'
    if np.array_equiv(x,[0,1,0,0]):
        return 'Class 3'
    if np.array_equiv(x,[1,0,0,0]):
        return 'Class 4'
    