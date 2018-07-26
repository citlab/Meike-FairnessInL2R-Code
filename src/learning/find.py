'''
Created on Jun 27, 2018

@author: meike.zehlike
'''
import numpy as np

def find_items_per_query(data, query_ids, which_query):
    """
    finds items which contains the given query_id
    :param data: all predictions or training judgments
    :param query_ids: list of query IDs
    :param which_query: given query ID
    :return: matrix filtered by which_query
    """
    return data[np.where(query_ids == which_query)[0], :]


def find_items_per_group_per_query(data, query_ids, which_query, prot_idx):
    """
    finds all the items with a given query ID and separates the items into protected
    and non-protected groups
    :param data: all predictions or training judgments
    :param query_ids: list of query IDs
    :param which_query: given query ID
    :param prot_idx: list states which item is protected or non-protected
    :return: three matrices
    """
    judgments_per_query = find_items_per_query(data, query_ids, which_query)
    prot_idx_per_query = find_items_per_query(prot_idx, query_ids, which_query)
    protected_items_per_query = judgments_per_query[np.where(prot_idx_per_query == True)[0], :]
    nonprotected_items_per_query = judgments_per_query[np.where(prot_idx_per_query == False)[0], :]

    return judgments_per_query, protected_items_per_query, nonprotected_items_per_query

