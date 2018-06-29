from Globals import *
from topp_prot import *
from topp import *
from exposure import exposure
from find import *

import numpy as np

def listnet_gradient(GAMMA, X, training_judgments, predictions, query_ids, prot_idx):

    # number of documents
    m = X.shape[0]
    # number of features
    p = X.shape[1]

    # find all training judgments and all predicted scores that belong to one query
    data_per_query = lambda which_query, data: \
                                   find_items_per_group_per_query(data, query_ids, which_query)

    # Exposure in rankings for protected and non-protected group
    exposure_prot_normalized = lambda which_query: exposure(data_per_query(which_query, predictions)[1],
                                                            data_per_query(which_query, predictions)[0])
    exposure_nprot_normalized = lambda which_query: exposure(data_per_query(which_query, predictions)[2],
                                                            data_per_query(which_query, predictions)[0])

    exposure_diff = lambda which_query : np.maximum(0, (exposure_nprot_normalized(which_query) - \
                                                        exposure_prot_normalized(which_query))) ** 2  # eq 5 from CIKM paper

    loss = lambda which_query:-np.sum(topp(data_per_query(which_query, training_judgments)[0]) *
                                          np.log(topp(data_per_query(which_query, predictions)[0])))  # eq 2 from CIKM paper

    # find all data points that belong to one query
    lx = lambda i: X[np.where(query_ids == query_ids[i]), :]

    # get idx of protected candidates per query, otherwise dimensions don't fit
    prot_idx_per_query = lambda i: prot_idx[np.where(query_ids == query_ids[i]), :]

    # returns only those lines in which the logical array idx contains 1
    l_group_vec = lambda preds, idx: preds[idx]
    l_group_mat = lambda data, idx: data[idx, :]

    tp1 = lambda t, u: np.dot(np.repeat(np.exp(u), len(t)))
    tp2 = lambda v: np.sum(np.exp(v))
    tp3 = lambda w, v: np.sum(np.dot(w, np.exp(v)))
    tp4 = lambda v: np.sum(np.exp(v)) ** 2

    # collect data structures
    group_features_p = lambda i: l_group_mat(lx(i), prot_idx_per_query(i))
    group_preds_p = lambda i: l_group_vec(lz(i), prot_idx_per_query(i))
    group_features_np = lambda i: l_group_mat(lx(i), ~prot_idx_per_query(i))
    group_preds_np = lambda i: l_group_vec(lz(i), ~prot_idx_per_query(i))

    tp_p = lambda i: np.sum((tp1(group_features_p(i)), group_preds_p(i)) * tp2(lz(i)) - np.exp(group_preds_p(i)) * tp3(lx(i), lz(i)) / tp4(lz(i)))
    tp_np = lambda i: np.sum((tp1(group_features_p(i)), group_preds_np(i)) * tp2(lz(i)) - np.exp(group_preds_np(i)) * tp3(lx(i), lz(i)) / tp4(lz(i)))

    group_size_p = lambda i: group_preds_p(i).shape[0]
    group_size_np = lambda i: group_preds_np(i).shape[0]

    # Exposure in Rankings for the protected and non-protected group
    exposure_prot = lambda i: np.sum(topp_prot(group_preds_p(i), lz(i))) / np.log(2)
    exposure_nprot = lambda i: np.sum(topp_prot(group_preds_np(i), lz(i))) / np.log(2)

    # normalize exposures
    exposure_prot_normalized = lambda i: exposure_prot(i) / group_size_p(i)
    exposure_nprot_normalized = lambda i: exposure_nprot(i) / group_size_np(i)

    if DEBUG_PRINT:
        print("exposure_prot_normalized:")
        print("exposure_nprot_normalized")

    u1 = lambda i: 2 * np.max((exposure_nprot_normalized(i) - exposure_prot_normalized(i)), 0)
    u2 = lambda i: (tp_np(i) / np.log(2)) / group_size_np(i)
    u3 = lambda i: (tp_p(i) / np.log(2)) / group_size_p(i)
    U = lambda i: u1(i) * (u2(i) - u3(i))

    l1 = lambda i: np.dot(lx(i), topp(ly(i)))
    l2 = lambda i: 1 / np.sum(np.exp(lz(i)))
    l3 = lambda i: np.dot(lx(i), np.exp(lz(i)))

    L = lambda i: l1(i) + l2(i) * l3(i)

    if ONLY_L:
        f = lambda i: L(i)

    if ONLY_U:
        f = lambda i: GAMMA * U(i)

    if L_AND_U:
        f = lambda i: GAMMA * U(i) + L(i)

    if DEBUG_PRINT:
        print("cost in gradient l")

    if DEBUG:
        iter = np.arange(m)
        prot_idx_q1 = prot_idx_per_query(1)
        lz1 = lz(1)
        lx1 = lx(1)

        z_prot = l_group_vec(lz(1), prot_idx_q1)
        z_nprot = l_group_vec(lz(1), ~prot_idx_q1)
        x_prot = l_group_mat(lx(1), prot_idx_per_query(1))

        topp_prot_p = topp_prot(z_prot, lz1)

        exposure_p = exposure_nprot(1)
        exposure_p_norm = exposure_nprot_normalized(1)

        tp1p = tp1(x_prot, z_prot)
        tp2p = tp2(lz(1))
        tp3p = tp3(lx(1), lz(1))
        twoMinusThree = tp2(lz(1)) - tp3(lx(1), lz(1))
        tp4p = tp4(lz(1))

        tp_prot = tp_p(1)

        u1expdiff = u1(1)
        u2np = u2(1)
        u3p = u3(1)

        fair_w = U(1)
        fair_w_gamma = GAMMA * U(1)
        acc_w = L(1)
        fval = fair_w_gamma + np.transpose(acc_w)

    grad = np.transpose(f(np.arange(m)).reshape(p, m))
