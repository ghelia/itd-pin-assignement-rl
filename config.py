import torch

class Config:

    ntypes = 3
    nitems = 10
    overlap_ratio = 0.25
    batch_size = 5

    items_emb_dim = 128
    items_dense_hidden_dim = 512
    items_nheads = 8
    items_nlayers = 3

    items_query_dense_hidden_dim = 512
    items_query_nheads = 8

    glimpse_dense_hidden_dim = 512
    glimpse_nheads = 8

    compatibility_emb = 128

    nodes_emb_dim = 128
    nodes_dense_hidden_dim = 512
    nodes_nheads = 8
    nodes_nlayers = 3

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    placed_flag_index = ntypes
    placement_offset = placed_flag_index + 1
    possible_neighbor_offset = placement_offset + ntypes
    item_dims = ntypes*2
    node_dims = ntypes + 1 + ntypes*2
