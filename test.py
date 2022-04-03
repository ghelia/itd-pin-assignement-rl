import torch
from attention_layer import *

batch_size = 7
nqueries = 9
nkvs = 11
emb_dim = 25
nhead = 5

encoder = AttentionEncoderLayer(emb_dim, 5)

qi = torch.rand([batch_size, nqueries, emb_dim]).to(Config.device)
kvi = torch.rand([batch_size, nkvs, emb_dim]).to(Config.device)
mask = torch.zeros([batch_size, nqueries, nkvs]).bool().to(Config.device)
mask[1,4,2] = True
mask[0,1,1] = True
mask[2,6,4] = True
outputs, attention = encoder(qi, kvi, mask=mask)

print(outputs.shape)
print(attention.shape)
print(attention[1])
