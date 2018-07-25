from learning import Globals
from learning import topp_prot
from learning import topp
from learning import exposure
from learning import find
import numpy as np
import multiprocessing
from joblib import Parallel, delayed

def listnet_gradient(GAMMA, training_features, training_judgments, predictions, query_ids, prot_idx):

    # number of documents
    m = training_features.shape[0]
    # number of features
    p = training_features.shape[1]
    # find all training judgments and all predicted scores that belong to one query
    data_per_query = lambda which_query, data: \
                                   find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
    # Exposure in rankings for protected and non-protected group
    #exposure_prot_normalized = lambda which_query: normalized_exposure.normalized_exposure(data_per_query(which_query, predictions)[1], \
                                                                     #data_per_query(which_query, predictions)[0])
    #exposure_nprot_normalized = lambda which_query: normalized_exposure.normalized_exposure(data_per_query(which_query, predictions)[2], \
                                                                    #data_per_query(which_query, predictions)[0])

    #u1 = lambda which_query: 2 * np.max((exposure_nprot_normalized(which_query) - exposure_prot_normalized(which_query)), 0)
    # u1 = lambda which_query: 2*normalized_exposure.exposure_diff(predictions, query_ids, which_query, prot_idx)
    # u2 = lambda which_query: topp_prot.normalized_topp_prot_deriv_per_group(data_per_query(which_query, training_features)[2], \
    #                                                    data_per_query(which_query, training_features)[0], \
    #                                                    data_per_query(which_query, predictions)[2], \
    #                                                    data_per_query(which_query, predictions)[0])  # derivative for non-protected group
    # u3 = lambda which_query: topp_prot.normalized_topp_prot_deriv_per_group(data_per_query(which_query, training_features)[1], \
    #                                                    data_per_query(which_query, training_features)[0], \
    #                                                    data_per_query(which_query, predictions)[1], \
    #                                                    data_per_query(which_query, predictions)[0])  # derivative for protected group

    # U_deriv = lambda which_query: u1(which_query) * (u2(which_query) - u3(which_query))

    U_deriv = lambda which_query: 2*exposure.exposure_diff(predictions, query_ids, which_query, prot_idx) * \
                                  topp_prot.normalized_topp_prot_deriv_per_group_diff(training_features, predictions, \
                                                                                      query_ids, which_query, prot_idx)

    ######asking Meike again because of the data structure#########
    l1 = lambda which_query: np.dot(np.transpose(data_per_query(which_query, training_features)[0]), topp.topp(data_per_query(which_query, training_judgments)[0]))
    l2 = lambda which_query: 1 / np.sum(np.exp(data_per_query(which_query, predictions)[0]))
    l3 = lambda which_query: np.dot(np.transpose(data_per_query(which_query, training_features)[0]), np.exp(data_per_query(which_query, predictions)[0]))

    L_deriv = lambda which_query: -l1(which_query) + l2(which_query) * l3(which_query)

    if Globals.ONLY_L:
        grad = lambda which_query: L_deriv(which_query)

    if Globals.ONLY_U:
        grad = lambda which_query: GAMMA * U_deriv(which_query)

    if Globals.L_AND_U:
        grad = lambda which_query: GAMMA * U_deriv(which_query) + L_deriv(which_query)


    # TODO: bin mir nicht sicher ob wir das hier brauchen...man müsste mal in octave gucken, was das gemacht hatte...
    # grad = np.transpose(f(np.arange(m)).reshape(p, m))

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(grad)(query) for query in np.unique(query_ids))

    return results
