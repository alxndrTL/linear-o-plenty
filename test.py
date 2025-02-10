import time
import torch
torch.set_float32_matmul_precision('high')

from linear import linear_attn, linear_attn_v2
from fla.ops.linear_attn.naive import naive_chunk_linear_attn
from ttt import naive_ttt, naive_ttt_lr_wd

device = 'cuda'
b, h, n, d = 16, 1, 512, 64

q = torch.randn(b, h, n, d, device=device)/float(d)**.5
k = torch.randn(b, h, n, d, device=device)/float(d)**.5
v = torch.randn(b, h, n, d, device=device)/float(d)
etas = torch.randn(b, h, n, device=device)
alphas = torch.randn(b, h, n, device=device)

linear_attn = torch.compile(linear_attn)
linear_attn_v2 = torch.compile(linear_attn_v2)
naive_chunk_linear_attn = torch.compile(naive_chunk_linear_attn)
naive_ttt = torch.compile(naive_ttt)
#naive_ttt_lr_wd = torch.compile(naive_ttt_lr_wd)

func = naive_ttt_lr_wd
iterations = 1000

# Warm-up
for _ in range(10):
    func(q, k, v, etas, alphas)

torch.cuda.synchronize()
start = time.time()
for _ in range(iterations):
    func(q, k, v, etas, alphas)
torch.cuda.synchronize()

print(f"Function: {func.__name__}")
print((time.time() - start) / iterations)

