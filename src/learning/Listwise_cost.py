import numpy as np
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
    data_per_query = lambda which_query, data: \
                                   find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)

    loss = lambda which_query:-np.dot(np.transpose(topp.topp(data_per_query(which_query, training_judgments)[0])),
                                          np.log(topp.topp(data_per_query(which_query, predictions)[0])))  # eq 2 from DELTR paper

    if Globals.ONLY_L:
        cost = lambda which_query: loss(which_query)

    if Globals.L_AND_U:
        cost = lambda which_query: GAMMA * exposure.exposure_diff(predictions, query_ids, which_query, prot_idx) ** 2 + loss(which_query)

    results = [cost(query) for query in query_ids]

    return np.asarray(results)
