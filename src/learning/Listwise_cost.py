import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from src.learning import Globals
from src.learning import topp
from src.learning import find
from src.learning import exposure

def listwise_cost(GAMMA, training_judgments, predictions, query_ids, prot_idx):

    # find all training judgments and all predicted scores that belong to one query
    ##print(query_ids)
    data_per_query = lambda which_query, data: \
                                   find.find_items_per_group_per_query(data, query_ids, which_query, prot_idx)
    #print("predictions")
    #print(predictions)
    # Exposure in rankings for protected and non-protected group
    exposure_prot_normalized = lambda which_query: exposure.exposure(data_per_query(which_query, predictions)[1],
                                                            data_per_query(which_query, predictions)[0])
    exposure_nprot_normalized = lambda which_query: exposure.exposure(data_per_query(which_query, predictions)[2],
                                                            data_per_query(which_query, predictions)[0])

    exposure_diff = lambda which_query : np.maximum(0, (exposure_nprot_normalized(which_query) - \
                                                        exposure_prot_normalized(which_query))) ** 2  # eq 5 from CIKM paper

    loss = lambda which_query:-np.sum(topp.topp(data_per_query(which_query, training_judgments)[0]) *
                                          np.log(topp.topp(data_per_query(which_query, predictions)[0])))  # eq 2 from CIKM paper

    if Globals.ONLY_L:
        cost = lambda which_query: loss(which_query)

    if Globals.L_AND_U:
        cost = lambda which_query: GAMMA * exposure_diff(which_query) + loss(which_query)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(cost)(query) for query in np.unique(query_ids))

    return results
