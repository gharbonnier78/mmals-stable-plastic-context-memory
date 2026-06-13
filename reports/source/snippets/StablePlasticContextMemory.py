class StablePlasticContextMemory(nn.Module):
    """Dataset-agnostic context-posterior fusion with exact legacy fallback."""
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.adaptive_gate = nn.Sequential(
            nn.Linear(cfg.metadata_dim, cfg.adaptive_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.adaptive_hidden_dim, 3),
        )
        prior = torch.tensor([cfg.prior_single, cfg.prior_plastic, cfg.prior_stable])
        final = self.adaptive_gate[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(prior.clamp_min(1e-8).log())

    def forward(
        self,
        *,
        q_single,
        q_legacy_prototype=None,
        q_stable=None,
        q_plastic=None,
        metadata=None,
        mode=None,
    ):
        active_mode = str(mode or self.cfg.mode)
        q_single = _spcm_normalize_probabilities(q_single)
        if active_mode == "legacy":
            if q_legacy_prototype is None:
                raise ValueError("legacy mode requires q_legacy_prototype")
            q_proto = _spcm_normalize_probabilities(q_legacy_prototype)
            w = float(self.cfg.legacy_single_weight)
            posterior = _spcm_normalize_probabilities(w * q_single + (1.0 - w) * q_proto)
            weights = posterior.new_tensor([w, 1.0 - w, 0.0]).expand(len(posterior), -1)
            return posterior, {
                "mode": active_mode,
                "fusion_weights_single_plastic_stable": weights,
                "posterior_entropy": _spcm_entropy(posterior),
                "branch_disagreement": (q_single - q_proto).abs().mean(dim=-1),
            }
        if q_stable is None or q_plastic is None:
            raise ValueError(f"{active_mode} mode requires stable and plastic branches")
        q_stable = _spcm_normalize_probabilities(q_stable)
        q_plastic = _spcm_normalize_probabilities(q_plastic)
        if not (q_single.shape == q_stable.shape == q_plastic.shape):
            raise ValueError("All context branches must have the same shape")
        prior = q_single.new_tensor([
            self.cfg.prior_single, self.cfg.prior_plastic, self.cfg.prior_stable,
        ]).expand(len(q_single), -1)
        if active_mode == "stable_plastic":
            weights = q_single.new_tensor([
                self.cfg.single_weight, self.cfg.plastic_weight, self.cfg.stable_weight,
            ]).expand(len(q_single), -1)
            raw_weights = weights
        elif active_mode == "adaptive":
            if metadata is None:
                raise ValueError("adaptive mode requires metadata")
            if metadata.shape != (q_single.shape[0], self.cfg.metadata_dim):
                raise ValueError(
                    f"metadata must have shape {(q_single.shape[0], self.cfg.metadata_dim)}, "
                    f"got {tuple(metadata.shape)}"
                )
            raw_weights = torch.softmax(self.adaptive_gate(metadata), dim=-1)
            rho = float(self.cfg.prior_floor)
            weights = rho * prior + (1.0 - rho) * raw_weights
        else:
            raise ValueError(f"Unsupported context-memory mode: {active_mode}")
        posterior = _spcm_normalize_probabilities(
            weights[:, 0:1] * q_single
            + weights[:, 1:2] * q_plastic
            + weights[:, 2:3] * q_stable
        )
        return posterior, {
            "mode": active_mode,
            "fusion_weights_single_plastic_stable": weights,
            "raw_fusion_weights": raw_weights,
            "posterior_entropy": _spcm_entropy(posterior),
            "stable_plastic_disagreement": (q_stable - q_plastic).abs().mean(dim=-1),
            "single_winner": q_single.argmax(dim=-1),
            "plastic_winner": q_plastic.argmax(dim=-1),
            "stable_winner": q_stable.argmax(dim=-1),
            "deployed_winner": posterior.argmax(dim=-1),
        }
