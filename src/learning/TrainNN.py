from learning import Listnet_gradient
from learning import Listwise_cost
from learning import Globals
import numpy as np
import matplotlib.pyplot as plt


def trainNN(GAMMA, directory, list_id, X, y, T, e, quiet=False):
    """
    trains the Neural Network to find the optimal loss in listwise learning to rank

    :param GAMMA: a float parameter tuning the disparate exposure metric
    :param directory: directory for saving the plots
    :param list_id: list of query IDs
    :param X: training features
    :param y: training judgments
    :param T: iteration steps
    :param e: learning rate
    :param quiet:
    """
    m = X.shape[0]
    n_features = X.shape[1]
    n_lists = np.unique(list_id).shape[0]

    prot_idx = np.reshape(X[:, Globals.PROT_COL] == np.repeat(Globals.PROT_ATTR, m), (m, 1))
    # linear neural network parameter initialization
    omega = (np.random.rand(n_features, 1) * Globals.INIT_VAR).reshape(-1)

    cost_converge_J = np.zeros((T, 1))
    cost_converge_L = np.zeros((T, 1))
    cost_converge_U = np.zeros((T, 1))
    omega_converge = np.empty((T, n_features))

    for t in range(0, T):
        if quiet == False:
             print('iteration ', t)

        # forward propagation
        z = np.dot(X, omega)
        z = np.reshape(np.asarray(z).astype('float'), (len(z), 1))
        # cost
        if quiet == False:
            print('computing cost')

        # with regularization
        cost = Listwise_cost.listwise_cost(GAMMA, y, z, list_id, prot_idx)
        J = cost + np.transpose(np.multiply(z, z)) * Globals.LAMBDA
        cost_converge_J[t] = np.sum(J)

        if quiet == False:
            print("computing gradient")

        grad = Listnet_gradient.listnet_gradient(GAMMA, X, y, z, list_id, prot_idx)
        omega = omega - e * np.sum(np.asarray(grad)[0], axis=1)
        omega_converge[t, :] = np.transpose(omega[:])

        if quiet == False:
            print('\n')

    # plots
    plt.subplot(211)
    plt.plot(cost_converge_J)
    plt.subplot(212)
    plt.plot(omega_converge)
    plt.savefig(directory + 'cost_gradient.png')

    return omega
