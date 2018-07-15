'''
Created on Jun 27, 2018

@author: meike.zehlike
'''
import numpy as np

def find_items_per_query(data, query_ids, which_query):
    return data[np.where(query_ids == which_query)[0], :]


def find_items_per_group_per_query(data, query_ids, which_query, prot_idx):
    judgments_per_query = find_items_per_query(data, query_ids, which_query)
    prot_idx_per_query = find_items_per_query(prot_idx, query_ids, which_query)
    ##There must be a better solution to implement this --> reshape
    prot_idx_per_query= np.reshape(prot_idx_per_query,(prot_idx_per_query.shape[0],))
    protected_items_per_query = judgments_per_query[prot_idx_per_query]
    nonprotected_items_per_query = judgments_per_query[~prot_idx_per_query]

    return judgments_per_query, protected_items_per_query, nonprotected_items_per_query

