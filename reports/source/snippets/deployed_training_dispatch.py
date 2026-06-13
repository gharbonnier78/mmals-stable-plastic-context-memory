if str(variant_cfg.get("context_memory_mode", "legacy")) == "legacy":
                    # Exact historical training path for backward reproduction.
                    logits, cache = forward_by_method(
                        model, xb, task_id, config, prototypes=banks["legacy"], train=True,
                        noisy_context_id=noisy_context_id, y=yb, return_cache=True,
                    )
                else:
                    # Repaired candidates receive the main task signal through the
                    # same deployed context posterior used at inference.
                    tb_current = torch.full(
                        (len(yb),), int(task_id), dtype=torch.long, device=DEVICE
                    )
                    logits, cache = spcm_forward_deployed(
                        model, xb, config, context_memory, banks, variant_cfg, task_id,
                        true_task_ids=tb_current, force_oracle=False,
                    )
                task_loss = F.cross_entropy(logits, yb)
                fungal = (
                    LAMBDA_ENTROPY * entropy_loss(cache["routes"])
                    + LAMBDA_METABOLIC * metabolic_loss(cache["transformed"], cache["routes"])
                )
                context_calib, _ = context_calibration_loss(model, cache, task_id, config, epoch)
                single_replay, _ = context_replay_loss(
