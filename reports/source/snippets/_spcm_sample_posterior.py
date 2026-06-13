def _spcm_sample_posterior(values, prototypes, kind, temperature, n_contexts):
    energies = torch.full((len(values), n_contexts), 1e6, device=values.device, dtype=values.dtype)
    available = torch.zeros(n_contexts, dtype=torch.bool, device=values.device)
    for cid, proto in prototypes.items():
        cid = int(cid)
        if cid >= n_contexts or kind not in proto:
            continue
        energies[:, cid] = _spcm_sample_diag_energy(values, proto[kind])
        available[cid] = True
    if not available.any():
        q = torch.full_like(energies, 1.0 / max(n_contexts, 1))
        return q, torch.zeros(len(values), device=values.device), energies
    q = torch.softmax(-energies / max(float(temperature), 1e-6), dim=-1)
    if n_contexts > 1:
        top2 = torch.topk(energies, k=2, largest=False, dim=-1).values
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = torch.ones(len(values), device=values.device)
    return q, margin, energies
