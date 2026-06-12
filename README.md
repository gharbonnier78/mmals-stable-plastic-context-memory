# Stable-Plastic Context Memory for MMALS

Reviewer-facing design package for a dataset-agnostic, backward-compatible context-memory module.

## Core proposal

The module introduces three operating modes:

- `legacy`: exact historical context posterior and blend;
- `stable_plastic`: stable and plastic prototype banks with fixed audited fusion;
- `adaptive`: validation-calibrated fusion, deployed-posterior distillation and confident-collapse veto.

The proposal is motivated by the CORe50 v2.3 result: class coverage, forgetting and host diversity improve, but oldest-context top-1 remains zero.

## Package contents

- `paper/`: LaTeX source and compiled design note.
- `src/`: reference PyTorch interface, not yet integrated into the full MMALS notebook.
- `tests/`: compatibility and shape tests.
- `spec/`: mode and checkpoint schema.
- `docs/`: ablation plan, reviewer checklist and claim boundary.

## Build

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Test the reference module

```bash
pip install -r requirements.txt
pytest -q
```
