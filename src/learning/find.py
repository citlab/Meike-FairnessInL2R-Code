'''
Created on Jun 27, 2018

@author: mzehlike
'''
import numpy as np

def find_items_per_query(data, query_ids, which_query):
    return data[np.where(query_ids == query_ids[which_query]), :]
