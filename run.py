import torch
import tiktoken
import torch.nn as nn
from model import Transformer
from torch.nn import functional as F

enc = tiktoken.get_encoding("p50k_base")

n_vocab = enc.n_vocab

block_size = 32
batch_size = 64
n_embd = 2*32
n_heads = 2
n_blocks = 2
eval_iters = 200
learning_rate = 1e-4
max_iters = 7500
eval_interval = 500
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(block_size, n_vocab, n_embd, n_heads, n_blocks, dropout, device)
model.to(device)
model.load_state_dict(torch.load("", map_location=torch.device('cpu')))
model.eval()

while True:
    context = torch.tensor([enc.encode(input("> "))], dtype=torch.long, device=device)
    print(enc.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
