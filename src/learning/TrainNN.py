from src.learning  import Listnet_gradient
from src.learning import Listwise_cost
from src.learning import Globals
import numpy as np
import matplotlib.pyplot as plt

def trainNN(GAMMA, directory, list_id, X, y, T, e, quiet = False):
    m = X.shape[0]
    n_features = X.shape[1]
    n_lists = np.unique(list_id).shape[0]

    prot_idx = np.reshape(X[:,Globals.PROT_COL] == np.repeat(Globals.PROT_ATTR,m),(m,1))
    #linear neural network parameter initialzation
    omega = np.random.rand(n_features,1)*Globals.INIT_VAR

    cost_converge_J = np.zeros((T, 1))
    cost_converge_L = np.zeros((T, 1))
    cost_converge_U = np.zeros((T, 1))
    omega_converge = np.zeros((T, n_features))

    for t in range(0,T):
        if quiet == False:
            print('iteration ',t)

        #forward propagation
        z = np.dot(X,omega)

        #cost
        if quiet == False:
            print('computing cost')

        #with regularization
        cost = Listwise_cost.listwise_cost(GAMMA, y, z, list_id, prot_idx)
        J = cost + np.transpose(np.multiply(z,z))*Globals.LAMBDA
        cost_converge_J[t] = np.sum(J)

        if quiet == False:
            print("computing gradient")

        grad = Listnet_gradient.listnet_gradient(GAMMA, X, y, z, list_id, prot_idx)

        omega = omega - e*np.sum(grad)

        omega_converge[t, :] = omega[:]

        if quiet == False:
            print('\n')

    #plots
    plt.plot(cost_converge_J)
    plt.show()
    plt.plot(omega_converge)