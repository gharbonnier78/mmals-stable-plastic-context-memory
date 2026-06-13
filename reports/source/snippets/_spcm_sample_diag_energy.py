def _spcm_sample_diag_energy(values, proto_stats):
    mu = proto_stats["mean"].to(values.device, dtype=values.dtype)
    var = proto_stats["var"].to(values.device, dtype=values.dtype).clamp_min(
        float(CONTEXT_MEMORY_SAMPLE_VARIANCE_FLOOR)
    )
    # Dimension-normalized diagonal Gaussian NLL, without the constant term.
    return (((values - mu) ** 2) / var + var.log()).mean(dim=-1)
