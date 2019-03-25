import numpy as np

def simple_norm(x):
    return x/255

def simple_reshape(x,resolution=50):
    return (np.array(x).reshape(-1, resolution, resolution, 1))