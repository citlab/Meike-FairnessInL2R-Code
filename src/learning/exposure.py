'''
Created on Jun 29, 2018

@author: mzehlike
'''
import numpy as np
import topp_prot
import find

def normalized_exposure(group_data, all_data):
    '''
    calculates the exposure of a group in the entire ranking
    implementation of equation 4 in DELTR paper

    :param group_data: predictions of relevance scores for one group
    :param all_data: all predictions

    :return: float value that is normalized exposure in a ranking for one group
             nan if group size is 0
    '''
    return (np.sum(topp_prot.topp_prot(group_data, all_data) / np.log(2))) / group_data.size

def exposure_diff(data, query_ids, which_query, prot_idx):
    """
    computes the exposure difference between protected and non-protected groups
    implementation of equation 5 in DELTR paper but without the square

    :param data: all predictions
    :param query_ids: list of query IDs
    :param which_query: given query ID
    :param prot_idx: list states which item is protected or non-protected

    :return: float value
    """
    judgments_per_query, protected_items_per_query, nonprotected_items_per_query = \
        find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
    exposure_prot = normalized_exposure(protected_items_per_query, judgments_per_query)
    exposure_nprot = normalized_exposure(nonprotected_items_per_query, judgments_per_query)
    exposure_diff = np.maximum(0, (exposure_nprot - exposure_prot))

    return exposure_diff
