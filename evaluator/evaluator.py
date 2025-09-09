from collections import OrderedDict

import numpy as np
from sklearn.metrics import f1_score
from tabulate import tabulate
import torch

from .build_evaluator import EVALUATOR_REGISTRY

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat

def eval_func(distmat, q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0

    q_pids = q_pids.cpu().numpy()
    g_pids = g_pids.cpu().numpy()
    
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # compute cmc curve without filtering same camera view
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    rank1 = all_cmc[0] if all_cmc.size > 0 else 0.0

    return rank1, mAP


@EVALUATOR_REGISTRY.register()
class ReID:
    def __init__(self, cfg, len_query):
        self.len_query = len_query

        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, feats, aids):
        query_feats = feats[: self.len_query]
        gallery_feats = feats[self.len_query :]
        query_aids = aids[: self.len_query]
        gallery_aids = aids[self.len_query :]

        dist_mat = euclidean_distance(query_feats, gallery_feats)
        rank1, mAP = eval_func(dist_mat, query_aids, gallery_aids)
         
        self.display(rank1, mAP)
        return rank1, mAP


    def display(self, rank1, mAP):
        evaluation_table = [
            ["Queries #", f"{self.len_query:,}"],
            ["Rank1", f"{rank1:.2f}%"],
            ["mAP", f"{mAP:.2f}%"],
        ]
        print(tabulate(evaluation_table))


