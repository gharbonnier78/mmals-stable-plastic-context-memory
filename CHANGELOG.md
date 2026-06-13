# Changelog

## RC2O-v2.3.2

### Added

- Per-sample diagonal-Gaussian prototype posterior.
- Stationary input-coordinate stable prototype bank.
- Fixed stable-plastic fusion control.
- Adaptive fusion with explicit prior initialization and residual prior floor.
- Deployed-posterior task training for non-legacy candidates.
- Deployed-versus-best-branch gap gate.
- Maximum fusion-weight gate.
- Stable-bank oldest-context gate.
- Machine-readable tracked-change export.
- Versioned compact `.pt` manifest.

### Preserved

- Exact historical batch-Fréchet inference in `legacy_reference`.
- Dataset-agnostic interfaces and overlap-aware MNIST-family targets.
- Existing retention, forgetting, class-coverage and recent-task gates.
- Optional storage policy with `.pt` saving disabled by default.

### Motivation from RC2O-v2.3.1

The stable-plastic candidate improved old-context retention, weakest-task retention, class coverage and forgetting, but the adaptive gate assigned approximately 98.4% of its mass to `q_single`. The deployed context posterior scored 0.4788 while the best internal component branch scored 0.7692. Recent-two accuracy regressed by 0.0523, so the candidate was not promoted.
