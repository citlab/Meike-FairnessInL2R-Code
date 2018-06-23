import numpy as np

def topp_prot(u,v):
    return np.exp(u/np.sum(np.exp(v)))