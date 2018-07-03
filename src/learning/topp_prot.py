import numpy as np


def topp_prot(group_items, all_items):
    '''
    given a dataset of features what is the probability of being at the top position
    for one group (group_items) out of all items
    example: what is the probability of each female (or male respectively) item (group_items) to be in position 1
    '''
    return np.exp(group_items) / np.sum(np.exp(all_items))


def topp_prot_first_derivative(group_features, all_features, group_predictions, all_predictions):
    '''
    Derivative for topp_prot in pieces:
    group_features = feature vector of (non-) protected group
    group_predictions = predicted scores for (non-) protected group
    all_predictions = predictions of all data points
    all_features = feature vectors of all data points
    '''
    numerator1 = np.dot(np.repeat(np.exp(group_predictions), len(group_features)))
    numerator2 = np.sum(np.exp(all_predictions))
    numerator3 = np.sum(np.dot(all_features, np.exp(all_predictions)))
    denominator = np.sum(np.exp(all_predictions)) ** 2

    result = np.sum(numerator1 * numerator2 - np.exp(group_predictions) * numerator3) / denominator

    return result


def normalized_topp_prot_deriv_per_group(group_features, all_features, group_predictions, all_predictions):

    derivative = topp_prot_first_derivative(group_features, all_features, group_predictions, all_predictions)
    result = (derivative / np.log(2)) / group_predictions.size
    return result
