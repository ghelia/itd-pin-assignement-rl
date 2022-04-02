import torch
from torch.nn import MultiheadAttention as MHA


emb_dim = 100
nhead = 10
batch_size = 3
nqueries = 7
nkv = 5
mha = MHA(emb_dim, nhead, batch_first=True)

Q = torch.rand([batch_size, nqueries, emb_dim])
KV = torch.rand([batch_size, nkv, emb_dim])

mask = torch.zeros([batch_size, nqueries, nkv]).bool()
mask[1,4,2] = True
mask[0,1,1] = True
mask[2,6,4] = True
rmask = mask.repeat([1,nhead,1]).reshape([batch_size*nhead, nqueries, nkv])

outputs, attention = mha(Q, KV, KV, attn_mask=rmask)

print("outputs : ", outputs.shape)
print("attention : ", attention.shape)
print(attention)
