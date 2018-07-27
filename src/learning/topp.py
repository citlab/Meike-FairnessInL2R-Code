import numpy as np

def topp(v):
    """
    computes the probability of a document being
    in the first position of the ranking
    implementation of equation 7 in paper DELTR

    :param v: all training judgments or all predictions
    :return: float value which is a probability
    """
    return np.exp(v) / np.sum(np.exp(v))

