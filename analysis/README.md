# Analyses for the Management Science revision

All scripts run from the repository root. They use the repo's data loading, train/test split, and PRC code.
Set `OMP_NUM_THREADS=1` when running several in parallel (BLAS oversubscription makes them very slow).

| Script | What it answers | Output |
|---|---|---|
| `response_rounding_impact.py`, `run_rounding.sh` | How much the boundary-rounding issue in `compute_model_errors` (fixed on this branch) changes the paper's non-strategic / naive / ours comparison. `EXACT_RESPONSE=0` reproduces the old scoring. | `results/rounding_impact.jsonl` |
| `deployed_guarantee.py`, `run_all.sh`, `summarize_deployed.py` | OR AE critique 2: worst-group error of a single realized draw vs. the uniform mixture, the LP-sparsified mixture (support <= G+1), and the best single iterate. | `results/deployed_*.jsonl` (exact scoring), `results/repo_response/` (partial, old scoring) |
| `oracle_gap.py`, `run_oracle.sh` | OR AE critique 3: how far Robust PRC is from a stronger oracle (direction search with exact threshold sweep) at the same group weights, and what that does to Algorithm 1. | `results/oracle_gap.jsonl` |

Settings: Robust PRC with shift grid step 0.1 on [0, max_g tau_g], test_size 0.3, eta_t = t^(-1/2).
`deployed_guarantee.py` uses T = 3000 and 3 seeds; the other two use T = 1000 / 500 and one seed, so treat them as indicative.
