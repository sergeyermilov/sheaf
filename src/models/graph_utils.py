from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes


def k_hop_subgraph_dropout(
        node_idx: Union[int, List[int], Tensor],
        num_hops: int,
        edge_index: Tensor,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None,
        flow: str = 'source_to_target',
        directed: bool = False,
        hop_dropouts: Optional[Union[List[float], float]] = None,
        rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    if isinstance(hop_dropouts, float):
        hop_dropouts = [hop_dropouts] * num_hops
    elif hop_dropouts is None:
        hop_dropouts = [0] * num_hops

    subsets = [node_idx]

    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)

        dropout_proba = torch.zeros_like(edge_mask[edge_mask], dtype=torch.float16)
        dropout_proba[...] = hop_dropouts[hop]
        droped_edges = torch.bernoulli(dropout_proba, generator=rng).type(torch.bool)

        edge_mask[edge_mask.clone()] = ~droped_edges

        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes,), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask


def k_hop_subgraph_limit(
        node_idx: Union[int, List[int], Tensor],
        num_hops: int,
        edge_index: Tensor,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None,
        flow: str = 'source_to_target',
        directed: bool = False,
        hop_max_edges: Union[int, List[int]] = -1,
        rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    if isinstance(hop_max_edges, int):
        hop_max_edges = [hop_max_edges] * num_hops

    subsets = [node_idx]

    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)

        if hop_max_edges[hop] != -1:
            mask_ix = torch.argwhere(edge_mask)
            rand_ix = torch.randperm(mask_ix.shape[0], generator=rng)[:hop_max_edges[hop]]
            edge_mask[...] = False
            edge_mask[mask_ix[rand_ix]] = True

        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes,), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask