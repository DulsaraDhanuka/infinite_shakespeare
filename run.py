import torch
import tiktoken
import torch.nn as nn
from model import Transformer
from torch.nn import functional as F

enc = tiktoken.get_encoding("cl100k_base")

n_vocab = enc.n_vocab
block_size = 256
n_embd = 384
n_heads = 6
n_blocks = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Transformer(block_size, n_vocab, n_embd, n_heads, dropout, device)
model.to(device)
model.load_state_dict(torch.load("model-main.ckpt", map_location=torch.device('cpu')))
model.eval()

while True:
    context = torch.tensor([enc.encode(input("> "))], dtype=torch.long, device=device)
    print(enc.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
