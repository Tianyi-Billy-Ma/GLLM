import copy
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.utils import (
    k_hop_subgraph,
    subgraph,
    degree,
)
from torch_sparse import SparseTensor
from torch_geometric.data import Data


def permute_edges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    
    permute_num = int((edge_num - node_num) * aug_ratio)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()

    edge2remove_index = np.where(edge_index[1] < data.num_hyperedges[0].item())[0]
    edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges[0].item())[0]
    # edge2remove_index = np.where(edge_index[1] < data.num_hyperedges)[0]
    # edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges)[0]

    try:

        edge_keep_index = np.random.choice(
            edge2remove_index, (edge_num - node_num) - permute_num, replace=False
        )

    except ValueError:

        edge_keep_index = np.random.choice(
            edge2remove_index, (edge_num - node_num) - permute_num, replace=True
        )

    edge_after_remove1 = edge_index[:, edge_keep_index]
    edge_after_remove2 = edge_index[:, edge2keep_index]
    edge_index = np.concatenate((edge_after_remove1, edge_after_remove2), axis=1)
    data.edge_index = torch.tensor(edge_index)
    return data


def permute_hyperedges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_remove_index_dict = {ind: i for i, ind in enumerate(edge_remove_index)}

    edge_remove_index_all = [
        i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict
    ]
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove

    data.edge_index = torch.tensor(edge_index)

    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def adapt(data, aug_ratio, aug):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        index[he].append(i)
    # edge
    edge_index_orig = copy.deepcopy(data.edge_index)
    drop_weights = degree_drop_weights(data.edge_index, hyperedge_num)
    edge_index_1 = drop_edge_weighted(
        data.edge_index,
        drop_weights,
        p=aug_ratio,
        threshold=0.7,
        h=hyperedge_num,
        index=index,
    )

    # feature
    edge_index_ = data.edge_index
    node_deg = degree(edge_index_[0])
    feature_weights = feature_drop_weights(data.x, node_c=node_deg)
    x_1 = drop_feature_weighted(data.x, feature_weights, aug_ratio, threshold=0.7)
    if aug == "adapt_edge":
        data.edge_index = edge_index_1
    elif aug == "adapt_feat":
        data.x = x_1
    else:
        data.edge_index = edge_index_1
        data.x = x_1
    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p

    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.0

    return x


def degree_drop_weights(edge_index, h):
    edge_index_ = edge_index
    deg = degree(edge_index_[1])[:h]
    deg_col = deg
    s_col = torch.log(deg_col)
    weights = (s_col - s_col.min() + 1e-9) / (s_col.mean() - s_col.min() + 1e-9)
    return weights


def feature_drop_weights(x, node_c):
    x = torch.abs(x).to(torch.float32)
    # 100 x 2012 mat 2012-> 100
    w = x.t() @ node_c
    w = w.log() + 1e-7
    # s = (w.max() - w) / (w.max() - w.mean())
    s = (w - w.min()) / (w.mean() - w.min())
    return s


def drop_edge_weighted(
    edge_index, edge_weights, p: float, h, index, threshold: float = 1.0
):
    _, edge_num = edge_index.size()
    edge_weights = (edge_weights + 1e-9) / (edge_weights.mean() + 1e-9) * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    # keep probability
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    edge_remove_index = np.array(list(range(h)))[sel_mask.cpu().numpy()]
    edge_remove_index_all = []
    for remove_index in edge_remove_index:
        edge_remove_index_all.extend(index[remove_index])
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    return edge_index


def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.mean(dim=0)
    zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token

    return data



def aug(data, aug_type, aug_ratio=0.2):
    data_aug = copy.deepcopy(data)
    if aug_type == "mask":
        data_aug = mask_nodes(data_aug, aug_ratio)
        return data_aug
    elif aug_type == "edge":
        data_aug = permute_edges(data_aug, aug_ratio)
        return data_aug
    elif aug_type == "hyperedge":
        data_aug = permute_hyperedges(data_aug, aug_ratio)
    elif aug_type == "none":
        return data_aug
    elif "adapt" in aug_type:
        data_aug = adapt(data_aug, aug_ratio, aug_type)
    else:
        raise ValueError(f"not supported augmentation")
    return data_aug