from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes


"""
Very restricted version of k-hop sampling that requireds edges to be ordered
"""
def k_hop_subgraph_limit(
        node_idx: Union[int, List[int], Tensor, Tensor],
        num_hops: int,
        edge_index: Tensor,
        num_nodes: Optional[int] = None,
        hop_max_edges: Union[int, List[int]] = -1,
        rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    edge_mask_subsets = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    if isinstance(hop_max_edges, int):
        hop_max_edges = [hop_max_edges] * num_hops

    subsets = [node_idx]
    half = len(edge_mask) // 2

    assert torch.all(row[:half] == col[half:]) and torch.all(row[half:] == col[:half]), \
        "Edges should be stacked the following way [[u, i], [i, u]]"

    edge_mask_subsets.fill_(False)

    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True

        # fill the first half
        torch.index_select(node_mask, 0, row, out=edge_mask)

        if hop_max_edges[hop] != -1:
            mask_ix = torch.argwhere(edge_mask[:half] & ~edge_mask_subsets[:half])
            rand_ix = torch.randperm(mask_ix.shape[0], generator=rng, device=row.device)[:hop_max_edges[hop]]

            # fill the first half (sample restricted by size)
            edge_mask[...] = False
            edge_mask[mask_ix[rand_ix]] = True

        # fill the second half
        mask_ix = torch.argwhere(edge_mask)
        edge_mask[half + mask_ix] = True
        subsets.append(col[edge_mask])
        edge_mask_subsets |= edge_mask

    edge_index_all_hops = edge_index[:, edge_mask_subsets]

    return edge_index_all_hops, edge_mask_subsets


def compute_adj_normalized(adjacency_matrix):
    degree = adjacency_matrix.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    diag_degree_inv_sqrt = torch.diag(degree_inv_sqrt)
    return diag_degree_inv_sqrt @ adjacency_matrix @ diag_degree_inv_sqrt