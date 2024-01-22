import json
import os
import time
import torch
import tiktoken
import argparse
import torch.nn as nn
from model import Transformer
from torch.nn import functional as F
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(
                    prog='Infinite Shakespeare Trainer',
                    description='Train the transformer on a dataset')

parser.add_argument('-i', '--input', help='Input file', default='data/tiny-shakespeare.txt')
parser.add_argument('-o', '--output', help='Output directory', default='.')
parser.add_argument('--block_size', help='Block (context) size', type=int, required=True)
parser.add_argument('--batch_size', help='Batch size', type=int, required=True)
parser.add_argument('--embedding_size', help='Embedding dimensions', type=int, required=True)
parser.add_argument('--num_heads', help='Number of heads in the multi-head attention layer', type=int, required=True)
parser.add_argument('--num_blocks', help='Number of transformer blocks', type=int, required=True)
parser.add_argument('--eval_iters', help='Evaluation iterations', type=int, required=True)
parser.add_argument('--eval_interval', help='Evaluation interval', type=int, required=True)
parser.add_argument('--learning_rate', help='Learning rate', type=float, required=True)
parser.add_argument('--max_iters', help='Max Iterations', type=int, required=True)
parser.add_argument('--dropout', help='Dropout rate', type=float, required=True)
args = parser.parse_args()

enc = tiktoken.get_encoding("p50k_base")
with open(args.input, 'r') as f:
    text = f.read()
tokens = enc.encode(text)
n_vocab = enc.n_vocab

block_size = args.block_size
batch_size = args.batch_size
n_embd = args.embedding_size
n_heads = args.num_heads
n_blocks = args.num_blocks
eval_iters = args.eval_iters
learning_rate = args.learning_rate
max_iters = args.max_iters
eval_interval = args.eval_interval
dropout = args.dropout
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

model = Transformer(block_size, n_vocab, n_embd, n_heads, n_blocks, dropout, device)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

training_epoch_loss = []
validation_epoch_loss = []
model_id = time.time()
for step in range(max_iters):
    try:
        if step % eval_interval == 0:
            losses = estimate_loss()
            training_epoch_loss.append(losses['train'])
            validation_epoch_loss.append(losses['val'])
            torch.save(model.state_dict(), os.path.join(args.output, f"model-{model_id}-{step}.pth"))
            print(f"step: {step}, training loss: {losses['train']}, validation loss: {losses['val']}")
        logits, loss = model(*get_batch("train"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    except KeyboardInterrupt as e:
        break

losses = estimate_loss()
training_epoch_loss.append(losses['train'])
validation_epoch_loss.append(losses['val'])
torch.save(model.state_dict(), os.path.join(args.output, f"model-{model_id}-{step}.pth"))
print(f"step: {step}, training loss: {losses['train']}, validation loss: {losses['val']}")

with open(os.path.join(args.output, f"model-{model_id}.config"), 'w') as f:
    f.write(json.dumps(args))

plt.plot(training_epoch_loss, label='train_loss')
plt.plot(validation_epoch_loss,label='val_loss')
plt.legend()
plt.show()
