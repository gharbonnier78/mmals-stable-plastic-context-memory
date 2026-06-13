# RC2O-v2.3.2 implementation notes

## Backward-compatible modes

- `legacy`: historical batch-Fréchet posterior and historical single/prototype blend.
- `stable_plastic`: fixed blend of single, plastic and stable per-sample branches.
- `adaptive`: learned blend with explicit prior and residual prior floor.

## Stationary stable coordinates

The `stationary_input` bank stores diagonal statistics in the representation entering MMALS. For MNIST-family runs this is the fixed input vector. For external visual datasets it is the frozen visual feature vector. This avoids treating a prototype as stable while the learned coordinate system producing it continues to drift.

## Per-sample evidence

For a sample vector `x` and context statistics `(mu_c, var_c)`, the notebook uses a dimension-normalized diagonal Gaussian energy:

```text
E_c(x) = mean_j [ (x_j - mu_cj)^2 / var_cj + log(var_cj) ]
q(c|x) = softmax(-E_c(x) / temperature)
```

Unlike the v2.3.1 batch-Fréchet branch, this produces a distinct context posterior for each item in a mixed replay batch.

## Compact checkpoint policy

`V232_PT_SAVE_POLICY = "off"` is the default. Use `final` for one checkpoint per selected variant and seed. Replay memory remains excluded by default to limit package size.

## Static validation completed

- Notebook JSON parsed successfully.
- All 44 code cells passed Python syntax validation after excluding notebook shell/magic lines.
- No v2.3.2 training result is embedded or claimed.
