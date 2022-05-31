import numpy as np

import torch
from torch import nn

from graph.kbcr import BaseLatentFeatureModel

from typing import Tuple, Dict, List


def make_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


def evaluate(entity_embeddings: nn.Embedding,
             predicate_embeddings: nn.Embedding,
             all_triples,
             test_triples,
             model: BaseLatentFeatureModel,
             batch_size: int,
             device: torch.device,
             index2features=None):

    xs = test_triples[:,0]
    xp = test_triples[:,1]
    xo = test_triples[:,2]

    sp_to_o, po_to_s = {}, {}
    for s_idx, p_idx, o_idx in all_triples:
        # s_idx, p_idx, o_idx = entity_to_index.get(s), predicate_to_index.get(p), entity_to_index.get(o)
        sp_key = (s_idx, p_idx)
        po_key = (p_idx, o_idx)

        if sp_key not in sp_to_o:
            sp_to_o[sp_key] = []
        if po_key not in po_to_s:
            po_to_s[po_key] = []

        sp_to_o[sp_key] += [o_idx]
        po_to_s[po_key] += [s_idx]

    assert xs.shape == xp.shape == xo.shape
    nb_test_triples = xs.shape[0]

    batches = make_batches(nb_test_triples, batch_size)

    hits = dict()
    hits_at = [1, 3, 5, 10]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0

    def hits_at_n(n_, rank):
        if rank <= n_:
            hits[n_] = hits.get(n_, 0) + 1

    counter = 0
    mrr = 0.0

    ranks_l, ranks_r = [], []
    for start, end in batches:
        batch_xs = xs[start:end]
        batch_xp = xp[start:end]
        batch_xo = xo[start:end]

        batch_size = batch_xs.shape[0]
        counter += batch_size * 2

        with torch.no_grad():
            tensor_xs = torch.from_numpy(batch_xs).to(device)
            tensor_xp = torch.from_numpy(batch_xp).to(device)
            tensor_xo = torch.from_numpy(batch_xo).to(device)

            if index2features is not None:
                tensor_xs = index2features[tensor_xs]
                tensor_xo = index2features[tensor_xo]

            scores = model(tensor_xp, tensor_xs, tensor_xo)
            scores_sp, scores_po = scores[0].cpu().numpy(), scores[1].cpu().numpy()

        batch_size = batch_xs.shape[0]
        for elem_idx in range(batch_size):
            s_idx, p_idx, o_idx = batch_xs[elem_idx], batch_xp[elem_idx], batch_xo[elem_idx]

            # Code for the filtered setting
            sp_key = (s_idx, p_idx)
            po_key = (p_idx, o_idx)

            o_to_remove = sp_to_o[sp_key]
            s_to_remove = po_to_s[po_key]

            for tmp_o_idx in o_to_remove:
                if tmp_o_idx != o_idx:
                    scores_sp[elem_idx, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[elem_idx, tmp_s_idx] = - np.infty
            # End of code for the filtered setting

            rank_l = 1 + np.argsort(np.argsort(- scores_po[elem_idx, :]))[s_idx]
            rank_r = 1 + np.argsort(np.argsort(- scores_sp[elem_idx, :]))[o_idx]

            ranks_l += [rank_l]
            ranks_r += [rank_r]

            mrr += 1.0 / rank_l
            mrr += 1.0 / rank_r

            for n in hits_at:
                hits_at_n(n, rank_l)

            for n in hits_at:
                hits_at_n(n, rank_r)

    counter = float(counter)

    mrr /= counter

    for n in hits_at:
        hits[n] /= counter

    metrics = dict()
    metrics['MRR'] = mrr
    for n in hits_at:
        metrics['hits@{}'.format(n)] = hits[n]

    return metrics