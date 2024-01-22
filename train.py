import torch
import tiktoken
import torch.nn as nn
from model import Transformer
from torch.nn import functional as F

enc = tiktoken.get_encoding("cl100k_base")
with open('data/input.txt', 'r') as f:
    text = f.read()
tokens = enc.encode(text)
n_vocab = enc.n_vocab

block_size = 256
batch_size = 64
n_embd = 384
n_heads = 6
n_blocks = 6
eval_iters = 200
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.tensor(tokens, dtype=torch.long)
n = int(0.9*data.shape[0])
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = Transformer(block_size, n_vocab, n_embd, n_heads, dropout, device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {step}, training loss: {losses['train']}, validation loss: {losses['val']}")
    logits, loss = model(*get_batch("train"))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
