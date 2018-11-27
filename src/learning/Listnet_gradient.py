from learning import Globals
from learning import topp_prot
from learning import topp
from learning import exposure
from learning import find
import numpy as np


def listnet_gradient(gamma, training_features, training_judgments, predictions, query_ids, prot_idx):
    """
    finds the optimal solution for the listwise cost
    implementation of equation 8 and appendix  A in paper DELTR

    :param gamma: a float parameter tuning the disparate exposure metric
    :param training_features: containing all the features
    :param training_judgments: vector containing the training judgments/ scores
    :param predictions: vector containing the prediction scores
    :param query_ids: list of query IDs
    :param prot_idx: list stating which item is protected or non-protected
    :return: float value --> optimal listwise cost
    """
    # find all training judgments and all predicted scores that belong to one query
    data_per_query = lambda which_query, data: \
                                   find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
    # Exposure in rankings for protected and non-protected group
    U_deriv = lambda which_query: 2 * exposure.exposure_diff(predictions, query_ids, which_query, prot_idx) * \
                                  topp_prot.normalized_topp_prot_deriv_per_group_diff(training_features, predictions, \
                                                                                      query_ids, which_query, prot_idx)
    # Training error
    l1 = lambda which_query: np.dot(np.transpose(data_per_query(which_query, training_features)[0]), topp.topp(data_per_query(which_query, training_judgments)[0]))
    l2 = lambda which_query: 1 / np.sum(np.exp(data_per_query(which_query, predictions)[0]))
    l3 = lambda which_query: np.dot(np.transpose(data_per_query(which_query, training_features)[0]), np.exp(data_per_query(which_query, predictions)[0]))

    L_deriv = lambda which_query:-l1(which_query) + l2(which_query) * l3(which_query)

    if Globals.ONLY_L:
        grad = lambda which_query: L_deriv(which_query)

    if Globals.ONLY_U:
        grad = lambda which_query: gamma * U_deriv(which_query)

    if Globals.L_AND_U:
        grad = lambda which_query: gamma * U_deriv(which_query) + L_deriv(which_query)

    results = [grad(query) for query in query_ids]

    return np.asarray(results)
