from Listnet_gradient import *
from Listwise_cost import *
from Globals import *
import numpy as np

def trainNN(GAMMA, directory, list_id, X, y, T, e, quiet = False):
    m = X.shape[0]
    n_features = X.shape[1]
    n_lists = np.unique(list_id).shape[0]

    prot_idx= X[:,PROT_COL] == PROT_ATTR

    #linear neural network parameter initialzation
    omega = np.random.rand(n_features,1)*INIT_VAR

    cost_converge_J = np.zeros(T, 1)
    cost_converge_L = np.zeros(T, 1)
    cost_converge_U = np.zeros(T, 1)
    omega_converge = np.zeros(T, n_features)

    for t in range(0,T):
        if quiet == False:
            print('iteration ',t)

        #forward propagation
        z = np.dot(X,omega)

        #cost
        if quiet == False:
            print('computing cost')

        #with regularization
        cost = listwise_cost(GAMMA, y, z, list_id, prot_idx)
        J = cost + np.transpose(np.multiply(z,z))*LAMDA
        cost_converge_J[t] = np.sum(J)

        if quiet == False:
            print("computing gradient")

        grad = listnet_gradient(GAMMA, X, y, z, list_id, prot_idx)

        #nochmal nachfragen, weil grad ein transpose hat in octave und eine 2
        omega = omega - e*np.sum(grad)

        omega_converge[t, :] = omega[:]

        if quiet == False:
            print('\n')
    #hier wird geplottet