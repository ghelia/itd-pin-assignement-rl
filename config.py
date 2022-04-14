import torch

class Config:

    learning_rate = 0.0003
    learning_rate_decay = 0.999
    n_epoch = 10000
    n_episode = 10
    batch_size = 1000
    paired_test_alpha = 0.02

    ntypes = 3
    nitems = 3
    overlap_ratio = 0.25
    check_neighbors = True

    items_emb_dim = 64
    items_dense_hidden_dim = 128
    items_nheads = 8
    items_nlayers = 2

    items_query_dense_hidden_dim = 128
    items_query_nheads = 8

    glimpse_dense_hidden_dim = 128
    glimpse_nheads = 8

    compatibility_emb = 128

    nodes_emb_dim = 64
    nodes_dense_hidden_dim = 128
    nodes_nheads = 8
    nodes_nlayers = 2

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
