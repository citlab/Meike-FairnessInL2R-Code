'''
Created on Jun 27, 2018

@author: mzehlike
'''
import numpy as np

def find_items_per_query(data, list_of_query_ids):
    return lambda i: data[np.where(list_of_query_ids == list_of_query_ids[i]), :]
