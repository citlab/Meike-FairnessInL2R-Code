'''
Created on Jun 29, 2018

@author: mzehlike
'''
import numpy as np
from topp_prot import topp_prot

def exposure(group_data, all_data):
    # eq 4 in cikm paper
    return (np.sum(topp_prot(group_data, all_data) / np.log(2))) / group_data.size
