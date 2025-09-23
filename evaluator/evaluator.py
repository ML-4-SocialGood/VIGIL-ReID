import torch
import numpy as np

from collections import OrderedDict
from sklearn.metrics import f1_score
from tabulate import tabulate

from .build_evaluator import EVALUATOR_REGISTRY
from metrics.euclidean_dist import euclidean_dist
from metrics.mAP_cmc import mAP_cmc


@EVALUATOR_REGISTRY.register()
class R1_mAP:
    def __init__(self, cfg, num_query, max_rank = 50, reranking = False):
        self.cfg = cfg
        self.num_query = num_query
        self.max_rank = max_rank
        self.reranking = reranking

        self.feats = []
        self.aids = []
        self.camids = []
        self.domains = []
    
    def reset(self):
        self.feats = []
        self.aids = []
        self.camids = []
        self.domains = []
        
    def process(self, batch_output):
        feats, aids, camids, domains = batch_output
        self.feats.append(feats)
        self.aids.extend(aids)
        self.camids.extend(camids)
        self.domains.extend(domains)

    def evaluate(self, display_ranks = [1, 5, 10]):
        results = OrderedDict()
        features = torch.cat(self.feats, dim = 0)    # Shape: (num_query + num_gallery, output_dim)
        
        query_feats = features[:self.num_query, :]                # Shape: (num_query, output_dim)
        query_aids = np.asarray(self.aids[:self.num_query])       # Shape: (num_query,)
        query_camids = np.asarray(self.camids[:self.num_query])   # Shape: (num_query,)
        query_domains = np.asarray(self.domains[:self.num_query]) # Shape: (num_query,)

        gallery_feats = features[self.num_query:, :]              # Shape: (num_gallery, output_dim)
        gallery_aids = np.asarray(self.aids[self.num_query:])     # Shape: (num_gallery,)
        gallery_camids = np.asarray(self.camids[self.num_query:]) # Shape: (num_gallery,)
        gallery_domains = np.asarray(self.domains[self.num_query:]) # Shape: (num_gallery,)

        dist_mat = np.asarray(euclidean_dist(query_feats, gallery_feats))    # Shape: (num_query, num_gallery)
        # make sure only the same domain distances are considered

        # compute per-domain metrics (by query domain)
        per_domain_results = None
        if self.domains is not None:
            unique_query_domains = np.unique(query_domains)
            per_domain_results = OrderedDict()
            for dom in unique_query_domains:
                print(f"Evaluating domain: {self.cfg.DATASET.TARGET_DOMAINS[dom]}")
                dom_query_mask = (query_domains == dom)
                dom_gallery_mask = (gallery_domains == dom)
                if not np.any(dom_query_mask):
                    continue
                # print(f"  Found {np.sum(dom_query_mask)} queries and {np.sum(dom_gallery_mask)} galleries for domain {dom}")
                # print(f"  Unique query AIDs: {np.unique(query_aids[dom_query_mask])}")
                # print(f"  Unique gallery AIDs: {np.unique(gallery_aids[dom_gallery_mask])}")
                dom_dist = dist_mat[dom_query_mask, :][:, dom_gallery_mask]
                dom_query_aids = query_aids[dom_query_mask]
                dom_gallery_aids = gallery_aids[dom_gallery_mask]
                dom_query_camids = query_camids[dom_query_mask]
                dom_gallery_camids = gallery_camids[dom_gallery_mask]
                try:
                    dom_cmc, dom_mAP = mAP_cmc(dom_dist, dom_query_aids, dom_gallery_aids, dom_query_camids, dom_gallery_camids, self.max_rank)
                    per_domain_results[int(dom)] = (dom_cmc, dom_mAP)
                except (AssertionError, RuntimeError):
                    # No valid queries for this domain (no positives in gallery)
                    # Use zero CMC curve of length max_rank and zero mAP
                    per_domain_results[int(dom)] = (np.zeros(self.max_rank, dtype=float), 0.0)

        self.display(per_domain_results, display_ranks)

        return per_domain_results
    
    def display(self, results, display_ranks):
        evaluation_table = []
        evaluation_table.append(["Domain", "mAP", "Rank-1", "Rank-5", "Rank-10"])
        for domain, (cmc, mAP) in results.items():
            row = [self.cfg.DATASET.TARGET_DOMAINS[domain]]
            mAP_str = f"{mAP:.2%}"
            row.append(mAP_str)
            for r in display_ranks:
                # Ensure cmc is an array and has enough elements
                rank = f"{cmc[r - 1]:.2%}"
                row.append(rank)
            evaluation_table.append(row)
        print(tabulate(evaluation_table))
    

@EVALUATOR_REGISTRY.register()
class Classification:
    def __init__(self, cfg, class_label_name_mapping = None):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, model_output, ground_truth):
        pred = model_output.max(1)[1]
        matches = pred.eq(ground_truth).float()
        self._correct += int(matches.sum().item())
        self._total += ground_truth.shape[0]
        self._y_true.extend(ground_truth.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        accuracy = 100.0 * self._correct / self._total
        error_rate = 100.0 - accuracy
        macro_f1 = 100.0 * f1_score(
            self._y_true, self._y_pred, average = "macro", labels = np.unique(self._y_true)
        )

        results["accuracy"] = accuracy
        results["error_rate"] = error_rate
        results["macro_f1"] = macro_f1

        evaluation_table = [
            ["Total #", f"{self._total:,}"], 
            ["Correct #", f"{self._correct:,}"], 
            ["Accuracy", f"{accuracy:.2f}%"], 
            ["Error Rate", f"{error_rate:.2f}%"], 
            ["Macro_F1", f"{macro_f1:.2f}%"],
        ]
        print(tabulate(evaluation_table))

        return results
