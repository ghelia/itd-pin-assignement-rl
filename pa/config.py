import torch

class Config:

    learning_rate = 0.00003
    learning_rate_decay = 0.9995
    n_epoch = 10000
    n_episode = 100
    batch_size = 20
    paired_test_alpha = 0.02

    selection_policy_weight = 0.25

    ntypes = 5
    nitems = 4
    overlap_ratio = 0.33
    check_neighbors = True

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    placed_flag_index = ntypes
    placement_offset = placed_flag_index + 1
    possible_neighbor_offset = placement_offset + ntypes

    item_dims = ntypes*2
    items_emb_dim = 128
    node_dims = ntypes + 1 + ntypes*2
    nodes_emb_dim = 128
