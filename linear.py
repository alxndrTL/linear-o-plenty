import torch

def linear_attn(q, k, v):
    # q, k, v: (B, H, L, D)
    B, H, L, D = q.shape

    chunk_size = 64
    assert L % chunk_size == 0

    # (B, H, n, C, D) where n is the number of chunks and C their size
    q = q.view(B, H, L // chunk_size, chunk_size, D) * (D ** -0.5)
    k = k.view(B, H, L // chunk_size, chunk_size, D)
    v = v.view(B, H, L // chunk_size, chunk_size, D)

    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv
    intra = ((
        q @ k.transpose(-1, -2)).masked_fill_(
        torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
        0
    )) @ v
    o = inter + intra
    return o.view(B, H, L, D)

def linear_attn_v2(q, k, v):
    # q, k, v: (B, H, L, D)
    B, H, L, D = q.shape

    chunk_size = 64
    assert L % chunk_size == 0

    # (B, H, n, C, D) where n is the number of chunks and C their size
    q = q.view(B, H, L // chunk_size, chunk_size, D) * (D ** -0.5)
    k = k.view(B, H, L // chunk_size, chunk_size, D)
    v = v.view(B, H, L // chunk_size, chunk_size, D)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1)
    o = torch.zeros_like(q)
    h = torch.zeros(B, H, D, D, device=q.device)
    for i in range(L // chunk_size):
        o[:, :, i] = q[:, :, i] @ h + (q[:, :, i] @ k[:, :, i].transpose(-1, -2)).masked_fill_(mask, 0) @ v[:, :, i]
        h = h + k[:, :, i].transpose(-1, -2) @ v[:, :, i]
    
    return o.view(B, H, L, D)
