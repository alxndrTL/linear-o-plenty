import torch

def naive_ttt(q, k, v, eta=0.5):
    # q, k, v: (B, H, L, D)
    B, H, L, D = q.shape

    batch_size = 64
    assert L % batch_size == 0

    # (B, H, n, b, D) where n is the number of batches and b their size
    q = q.view(B, H, L // batch_size, batch_size, D) * (D ** -0.5)
    k = k.view(B, H, L // batch_size, batch_size, D)
    v = v.view(B, H, L // batch_size, batch_size, D)

    o = torch.zeros_like(q)
    h = torch.zeros(B, H, D, D, device=o.device)
    for i in range(L//batch_size):
        temp = k[:, :, i] @ h - v[:, :, i] # (B, H, b, D)
        o[:, :, i] = q[:, :, i] @ h - 2 * eta * ((q[:, :, i] @ k[:, :, i].transpose(-1, -2)).masked_fill_(torch.triu(torch.ones(batch_size, batch_size, dtype=bool, device=q.device), diagonal=1), 0)) @ temp
        h = h - 2 * eta * k[:, :, i].transpose(-1, -2) @ temp

    return o.view(B, H, L, D)

def naive_ttt_lr_wd(q, k, v, etas, alphas):
    # q, k, v: (B, H, L, D)
    # etas, betas: (B, H, L)
    B, H, L, D = q.shape

    batch_size = 64
    assert L % batch_size == 0

    # (B, H, n, b, D) where n is the number of batches and b their size
    q = q.view(B, H, L // batch_size, batch_size, D) * (D ** -0.5)
    k = k.view(B, H, L // batch_size, batch_size, D)
    v = v.view(B, H, L // batch_size, batch_size, D)
    etas = etas.view(B, H, L // batch_size, batch_size)
    alphas = alphas.view(B, H, L // batch_size, batch_size)
    
    o = torch.zeros_like(q)
    mask = torch.triu(torch.ones(batch_size, batch_size, dtype=torch.bool, device=q.device), diagonal=1)
    h = torch.zeros(B, H, D, D, device=o.device)
    for b in range(q.shape[2]):
        betas = torch.cumsum(((1-alphas[:, :, b]) + 1e-10).log(), dim=2).exp()
        beta_b = betas[:, :, -1]

        temp = (etas[:, :, b] / betas).unsqueeze(-1) * (k[:, :, b] @ h - v[:, :, b])
        o[:, :, b] = betas.unsqueeze(-1) * q[:, :, b] @ h - 2 * (betas.unsqueeze(-1)*(q[:, :, b] @ k[:, :, b].transpose(-1, -2)).masked_fill_(mask, 0)) @ temp # (B, H, C, D)
        print(beta_b.shape)
        print(h.shape)
        h = beta_b * h - 2 * beta_b * k[:, :, b].transpose(-1, -2) @ temp # (B, H, D, D)

    return o.view(B, H, L, D)