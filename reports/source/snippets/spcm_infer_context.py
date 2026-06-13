def spcm_infer_context(
    model,
    x,
    config,
    context_memory,
    legacy_bank,
    stable_bank,
    plastic_bank,
    variant_cfg,
    current_task,
    true_task_ids=None,
    force_oracle=False,
):
    if force_oracle:
        if true_task_ids is None:
            raise ValueError("Oracle diagnostic requires true task ids")
        q = F.one_hot(true_task_ids.long().clamp(0, N_CONTEXTS - 1), num_classes=N_CONTEXTS).float()
        return q, {
            "mode": "oracle_diagnostic", "q_single": q,
            "q_feature_plastic": q, "q_latent_plastic": q, "q_input_stable": q,
            "q_plastic": q, "q_stable": q, "q_deployed": q,
            "fusion_weights": torch.tensor([1.0, 0.0, 0.0], device=x.device).expand(len(x), -1),
            "collapse_veto_applied": torch.zeros(len(x), dtype=torch.bool, device=x.device),
        }

    mode = str(variant_cfg.get("context_memory_mode", "legacy"))
    if mode == "legacy":
        # Exact historical path: batch-level Fréchet evidence and historical blend.
        legacy_branches = _spcm_proto_branch(
            model, x, legacy_bank, config,
            evidence_mode="batch_frechet_legacy",
            coordinate_mode="dynamic_feature_latent",
        )
        q_deployed, legacy_info = infer_batch_context_probs(model, x, legacy_bank, config=config)
        w = float(CONTEXT_MEMORY_LEGACY_SINGLE_WEIGHT)
        weights = q_deployed.new_tensor([w, 1.0 - w, 0.0]).expand(len(q_deployed), -1)
        return q_deployed, {
            "mode": "legacy", "q_single": legacy_branches["q_single"],
            "q_feature_plastic": legacy_branches["q_feature"],
            "q_latent_plastic": legacy_branches["q_latent"],
            "q_input_stable": legacy_branches["q_input"],
            "q_plastic": legacy_branches["q_proto"],
            "q_stable": legacy_branches["q_proto"],
            "q_deployed": q_deployed, "fusion_weights": weights,
            "collapse_veto_applied": torch.zeros(len(x), dtype=torch.bool, device=x.device),
            **legacy_info,
        }

    evidence_mode = str(variant_cfg.get("prototype_evidence_mode", "sample_diag_gaussian"))
    stable_coordinate_mode = str(variant_cfg.get("stable_coordinate_mode", "dynamic_feature_latent"))
    plastic = _spcm_proto_branch(
        model, x, plastic_bank, config,
        evidence_mode=evidence_mode,
        coordinate_mode="dynamic_feature_latent",
    )
    stable = _spcm_proto_branch(
        model, x, stable_bank, config,
        evidence_mode=evidence_mode,
        coordinate_mode=stable_coordinate_mode,
    )
    metadata = _spcm_metadata(
        plastic["q_single"], plastic["q_proto"], stable["q_proto"], stable_bank, current_task
    )
    q_deployed, fusion_audit = context_memory(
        q_single=plastic["q_single"], q_stable=stable["q_proto"],
        q_plastic=plastic["q_proto"], metadata=metadata, mode=mode,
    )
    veto_mask = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    if bool(variant_cfg.get("collapse_veto", False)):
        q_deployed, veto_mask = _spcm_apply_confident_recency_veto(
            q_deployed, stable["q_proto"], true_task_ids
        )
    return q_deployed, {
        "mode": mode, "q_single": plastic["q_single"],
        "q_feature_plastic": plastic["q_feature"],
        "q_latent_plastic": plastic["q_latent"],
        "q_input_stable": stable["q_input"],
        "q_plastic": plastic["q_proto"], "q_stable": stable["q_proto"],
        "q_deployed": q_deployed,
        "fusion_weights": fusion_audit["fusion_weights_single_plastic_stable"],
        "raw_fusion_weights": fusion_audit.get("raw_fusion_weights"),
        "metadata": metadata, "collapse_veto_applied": veto_mask,
        "feature_margin": plastic["feature_margin"],
        "latent_margin": plastic["latent_margin"],
        "input_margin": stable["input_margin"],
    }
