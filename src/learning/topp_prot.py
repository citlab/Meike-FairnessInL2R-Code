import numpy as np
from learning import find


def topp_prot(group_items, all_items):
    '''
    given a dataset of features what is the probability of being at the top position
    for one group (group_items) out of all items
    example: what is the probability of each female (or male respectively) item (group_items) to be in position 1
    implementation of equation 7 in paper DELTR

    :param group_items: vector of predicted scores of one group (protected or non-protected)
    :param all_items: vector of predicted scores of all items

    :return: numpy array of float values
    '''
    return np.exp(group_items) / np.sum(np.exp(all_items))


def topp_prot_first_derivative(group_features, all_features, group_predictions, all_predictions):
    '''
    Derivative for topp_prot in pieces:
    implementation of equation 11 in paper DELTR

    :param group_features: feature vector of (non-) protected group
    :param group_predictions: predicted scores for (non-) protected group
    :param all_predictions: predictions of all data points
    :param all_features: feature vectors of all data points

    :return: numpy array with weight adjustments
    '''
    # numerator1 = np.dot(group_features, np.repeat(np.exp(group_predictions), group_features.shape[0]))
    numerator1 = np.dot(np.transpose(np.exp(group_predictions)), group_features)
    numerator2 = np.sum(np.exp(all_predictions))
    numerator3 = np.sum(np.dot(np.transpose(np.exp(all_predictions)), all_features))
    denominator = np.sum(np.exp(all_predictions)) ** 2

    result = (numerator1 * numerator2 - np.exp(group_predictions) * numerator3) / denominator

    # return result as flat numpy array instead of matrix
    return np.asarray(result).reshape(-1)


def normalized_topp_prot_deriv_per_group(group_features, all_features, group_predictions, all_predictions):
    """
    normalizes the results of the derivative of topp_prot

    :param group_features: feature vector of (non-) protected group
    :param all_features: feature vectors of all data points
    :param group_predictions: predictions of all data points
    :param all_predictions: predictions of all data points

    :return: numpy array of float values
    """
    derivative = topp_prot_first_derivative(group_features, all_features, group_predictions, all_predictions)
    result = (derivative / np.log(2)) / group_predictions.size
    return result

def normalized_topp_prot_deriv_per_group_diff(training_features, predictions, query_ids, which_query, prot_idx):
    """
    calculates the difference of the normalized topp_prot derivative of the protected and non-protected groups
    implementation of the second factor equation 12 in paper DELTR

    :param training_features: vector of all features
    :param predictions: predictions of all data points
    :param query_ids: list of query IDs
    :param which_query: given query
    :param prot_idx: list stating which item is protected or non-protected

    :return: numpy array of float values
    """
    train_judgments_per_query, train_protected_items_per_query, train_nonprotected_items_per_query = \
        find.find_items_per_group_per_query(training_features, query_ids, which_query, prot_idx)

    predictions_per_query, pred_protected_items_per_query, pred_nonprotected_items_per_query = \
        find.find_items_per_group_per_query(predictions, query_ids, which_query, prot_idx)

    u2 = normalized_topp_prot_deriv_per_group(\
        train_nonprotected_items_per_query, \
        train_judgments_per_query, \
        pred_nonprotected_items_per_query, \
        predictions_per_query)  # derivative for non-protected group
    u3 = normalized_topp_prot_deriv_per_group(\
        train_protected_items_per_query, \
        train_judgments_per_query, \
        pred_protected_items_per_query, \
        predictions_per_query)  # derivative for protected group

    return u2 - u3

