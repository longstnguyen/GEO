import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_sparse import SparseTensor
from .baselines import Mahalanobis, ODIN


def rand_splits(node_idx, train_prop=0.5, valid_prop=0.25):
    """randomly splits label into train/valid/test splits"""
    splits = {}
    n = node_idx.size(0)

    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    splits["train"] = node_idx[train_indices]
    splits["valid"] = node_idx[val_indices]
    splits["test"] = node_idx[test_indices]

    return splits


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """converts the edge_index into SparseTensor"""
    num_edges = edge_index.size(1)

    (row, col), N, _E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError(
            "cumsum was found to be unstable: its last element does not correspond to sum"
        )
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.asarray(_pos).reshape((-1, 1))
    neg = np.asarray(_neg).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_pred.shape == y_true.shape:
        y_pred = y_pred.detach().cpu().numpy()
    else:
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list) / len(rocauc_list)


@torch.no_grad()
def evaluate_classify(model, dataset, eval_func, criterion, args, device):
    model.eval()

    train_idx, valid_idx, test_idx = (
        dataset.splits["train"],
        dataset.splits["valid"],
        dataset.splits["test"],
    )
    y = dataset.y
    out = model(dataset, device).cpu()

    train_score = eval_func(y[train_idx], out[train_idx])
    valid_score = eval_func(y[valid_idx], out[valid_idx])
    test_score = eval_func(y[test_idx], out[test_idx])

    if args.method != "GPN":
        if args.dataset in ("proteins", "ppi"):
            valid_loss = criterion(out[valid_idx], y[valid_idx].to(torch.float))
        else:
            valid_out = F.log_softmax(out[valid_idx], dim=1)
            valid_loss = criterion(valid_out, y[valid_idx].squeeze(1))

        return train_score, valid_score, test_score, valid_loss
    else:
        return train_score, valid_score, test_score


# @torch.no_grad()
def evaluate_detect(
    model, dataset_ind, dataset_ood, criterion, eval_func, args, device, return_score=False
):
    model.eval()

    if isinstance(model, Mahalanobis):
        test_ind_score = model.detect(
            dataset_ind,
            dataset_ind.splits["train"],
            dataset_ind,
            dataset_ind.splits["test"],
            device,
            args,
        )
    elif isinstance(model, ODIN):
        test_ind_score = model.detect(dataset_ind, dataset_ind.splits["test"], device, args).cpu()
    else:
        with torch.no_grad():
            test_ind_score = model.detect(
                dataset_ind, dataset_ind.splits["test"], device, args
            ).cpu()
    if isinstance(dataset_ood, list):
        result = []
        cohort_scores = []
        for d in dataset_ood:
            if isinstance(model, Mahalanobis):
                test_ood_score = model.detect(
                    dataset_ind, dataset_ind.splits["train"], d, d.node_idx, device, args
                ).cpu()
            elif isinstance(model, ODIN):
                test_ood_score = model.detect(d, d.node_idx, device, args).cpu()
            else:
                with torch.no_grad():
                    test_ood_score = model.detect(d, d.node_idx, device, args).cpu()
            auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
            result += [auroc] + [aupr] + [fpr]
            cohort_scores.append(test_ood_score)
    else:
        if isinstance(model, Mahalanobis):
            test_ood_score = model.detect(
                dataset_ind,
                dataset_ind.splits["train"],
                dataset_ood,
                dataset_ood.node_idx,
                device,
                args,
            ).cpu()
        elif isinstance(model, ODIN):
            test_ood_score = model.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()
        else:
            with torch.no_grad():
                test_ood_score = model.detect(dataset_ood, dataset_ood.node_idx, device, args).cpu()
        auroc, aupr, fpr, _ = get_measures(test_ind_score, test_ood_score)
        result = [auroc] + [aupr] + [fpr]

    with torch.no_grad():
        out = model(dataset_ind, device).cpu()
    test_idx = dataset_ind.splits["test"]
    test_score = eval_func(dataset_ind.y[test_idx], out[test_idx])

    # valid_loss = 0.
    # for i, (d_in, d_out) in enumerate(zip(valid_loader_ind, valid_loader_ood)):
    #     valid_loss += model.loss_compute(d_in, d_out, criterion, device, args)
    # valid_loss /= (i+1)
    valid_idx = dataset_ind.splits["valid"]
    if args.dataset in ("proteins", "ppi"):
        valid_loss = criterion(out[valid_idx], dataset_ind.y[valid_idx].to(torch.float))
    else:
        valid_out = F.log_softmax(out[valid_idx], dim=1)
        valid_loss = criterion(valid_out, dataset_ind.y[valid_idx].squeeze(1))

    result += [test_score, valid_loss.detach().item()]

    if return_score:
        if isinstance(dataset_ood, list):
            test_ood_score = torch.cat(cohort_scores)
        return result, test_ind_score, test_ood_score
    else:
        return result


dataset_drive_url = {
    "snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia",
    "pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y",
    "yelp-chi": "1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ",
}

splits_drive_url = {
    "snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N",
    "pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_",
}
