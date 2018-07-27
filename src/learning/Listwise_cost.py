import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from learning import Globals
from learning import topp
from learning import find
from learning import exposure

def listwise_cost(GAMMA, training_judgments, predictions, query_ids, prot_idx):
    """
    computes the loss in list-wise learning to rank
    it incorporates L which is the error between the training judgments and those
    predicted by a model and U which is the disparate exposure metric
    implementation of equation 6 in DELTR paper

    :param GAMMA: a float parameter tuning the disparate exposure metric
    :param training_judgments: containing the training judgments/ scores
    :param predictions: containing the predicted scores
    :param query_ids: list of query IDs
    :param prot_idx: list stating which item is protected or non-protected
    :return: a float value --> loss
    """
    # find all training judgments and all predicted scores that belong to one query
    data_per_query = lambda which_query, data: \
                                   find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
    # Exposure in rankings for protected and non-protected group
    # exposure_prot_normalized = lambda which_query: normalized_exposure.normalized_exposure(data_per_query(which_query, predictions)[1],
                                                            # data_per_query(which_query, predictions)[0])
    # exposure_nprot_normalized = lambda which_query: normalized_exposure.normalized_exposure(data_per_query(which_query, predictions)[2],
                                                            # data_per_query(which_query, predictions)[0])

    # exposure_diff = lambda which_query : np.maximum(0, (exposure_nprot_normalized(which_query) - \
                                                        # exposure_prot_normalized(which_query))) ** 2  # eq 5 from CIKM paper

    loss = lambda which_query:-np.dot(np.transpose(topp.topp(data_per_query(which_query, training_judgments)[0])),
                                          np.log(topp.topp(data_per_query(which_query, predictions)[0])))  # eq 2 from CIKM paper

    if Globals.ONLY_L:
        cost = lambda which_query: loss(which_query)

    if Globals.L_AND_U:
        # cost = lambda which_query: GAMMA * exposure_diff(which_query) + loss(which_query)
        cost = lambda which_query: GAMMA * exposure.exposure_diff(predictions, query_ids, which_query, prot_idx) ** 2 + loss(which_query)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(cost)(query) for query in np.unique(query_ids))

    return results
