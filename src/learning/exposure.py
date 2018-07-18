'''
Created on Jun 29, 2018

@author: mzehlike
'''
import numpy as np
from src.learning import topp_prot
from src.learning import find

def exposure(group_data, all_data):
    # eq 4 in cikm paper
    return (np.sum(topp_prot.topp_prot(group_data, all_data) / np.log(2))) / group_data.size

def exposure_diff(data, query_ids, which_query, prot_idx):
    judgments_per_query, protected_items_per_query, nonprotected_items_per_query = \
        find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
    exposure_prot_normalized = exposure(protected_items_per_query, judgments_per_query)
    exposure_nprot_normalized = exposure(nonprotected_items_per_query, judgments_per_query)
    exposure_diff = np.maximum(0, (exposure_nprot_normalized - exposure_prot_normalized))

    return exposure_diff
