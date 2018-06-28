import numpy as np
from learning.Globals import *
from learning.topp_prot import *
from learning.topp import *
from learning.find import *

def listwise_cost(GAMMA, y, z, query_ids, prot_idx):

    # find all training judgments and all predicted scores that belong to one query
    ly = lambda i: y[np.where(query_ids == query_ids[i]), :]
    lz = lambda i: z[np.where(query_ids == query_ids[i]), :]

    # get idx of protected candidates per query, otherwise dimensiions don't fit
    prot_idx_per_query = lambda i: prot_idx[np.where(query_ids == query_ids[i]), :]
    l_prot_vec = lambda preds, idx: preds[idx]
    group_size_p = lambda i: l_prot_vec(lz(i), prot_idx_per_query(i)).shape[0]
    group_size_np = lambda i: l_prot_vec(lz(i), ~prot_idx_per_query[i]).shape[0]

    # Exposure in rankings for protected and non-protected group
    exposure_prot = lambda i: np.sum(topp_prot(l_prot_vec(lz(i), prot_idx_per_query(i)), lz(i)) / np.log(2))
    exposure_prot_normalized = lambda i: exposure_prot(i) / group_size_p(i)

    exposure_nprot = lambda i: np.sum(topp_prot(l_prot_vec(lz(i), ~prot_idx_per_query(i)), lz(i) / np.log(2)))
    exposure_nprot_normalized = lambda i: exposure_nprot(i) / group_size_np(i)

    exposure_diff = lambda i: (np.maximum(0, (exposure_nprot_normalized(i) - exposure_prot_normalized(i)))) ** 2

    accuracy = lambda i:-np.sum(topp(ly(i)) * np.log(topp(lz(i))))
    if (DEBUG):
        iter = 1
        idx = prot_idx_per_query(iter)
        z_prot = l_prot_vec(lz(iter), prot_idx_per_query(iter))
        z_nprot = l_prot_vec(lz(iter), ~prot_idx_per_query(iter))

        topl_prot = topp_prot(l_prot_vec(lz(iter), prot_idx_per_query(iter)), lz(iter))
        top1_nprot = topp_prot(l_prot_vec(lz(iter), ~prot_idx_per_query(iter)), lz(iter))
        top1_prot_times_v = topp_prot(l_prot_vec(lz(iter), prot_idx_per_query(iter)), lz(iter)) / np.log(2)
        top1_nprot_times_v = topp_prot(l_prot_vec(lz(iter), ~prot_idx_per_query(iter)), lz(iter)) / np.log(2)

        group_size_prot = group_size_p(iter)
        group_size_nprot = group_size_np(iter)

        exposure_p = exposure_prot(iter)
        exposure_p_norm = exposure_prot_normalized(iter)

        exposure_np = exposure_nprot(iter)
        exposure_np_norm = exposure_nprot_normalized(iter)

        exposure_difference = exposure_diff(iter)
        accuracy2 = accuracy(iter)

        cost = GAMMA * exposure_diff(iter) + accuracy(iter)

    if (ONLY_U):
        j = lambda i: GAMMA * exposure_diff(i)

    if (ONLY_L):
        j = lambda i: accuracy(i)

    if L_AND_U:
        u = lambda i: GAMMA * exposure_diff(i)
        l = lambda i: accuracy(i)
        j = lambda i: GAMMA * exposure_diff(i) + accuracy

    # J feht noch, wegen pararrayfun
    # FIXME: reimplement this such that you don't iterate over each element but over each query id
    J = j(np.arange(len(z)))
