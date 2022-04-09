import torch

class Config:

    ntypes = 8
    nitems = 20
    overlap_ratio = 0.25
    batch_size = 10

    items_emb_dim = 128
    items_dense_hidden_dim = 512
    items_nheads = 8
    items_nlayers = 3

    items_query_dense_hidden_dim = 512
    items_query_nheads = 8

    nodes_emb_dim = 128
    nodes_dense_hidden_dim = 512
    nodes_nheads = 8
    nodes_nlayers = 3

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
