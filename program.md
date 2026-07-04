# Cosmological Parameter Calibration

**Task:** Find cosmological parameters θ = {om, ob, h, ns, as_, w0} that reproduce the observed matter power spectrum in `obs_pk.npy`. Your benchmark score is the simulator call index at convergence — minimize it.

## Input files

| File | Description |
|------|-------------|
| `obs_pk.npy` | NumPy array shape (50,) — P(k) at k = logspace(−2, 0, 50) h/Mpc |
| `config/prior_bounds.yaml` | Parameter bounds, convergence threshold (`chi2.epsilon`), call budget (`budget.max_calls`) |

## Tools

```bash
python TOOLS_PATH/compute_chi2.py --params '{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}' --notes "reasoning"
```
→ prints `chi2=<value>  call_idx=<N>`, appends a row to `runs.csv`.

```bash
python TOOLS_PATH/get_pk.py --params '{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}'
```
→ prints JSON with keys `k`, `pk`, `obs_pk`, `residual_frac`. Also appends to `runs.csv`.

Both tools enforce prior bounds and error on out-of-range parameters.

## Scoring

```
chi2 = mean_k [ (P(θ, k) - P_obs(k))^2 / σ(k)^2 ]
```
where σ(k) = 0.02 × P_obs(k). Convergence threshold ε is in `config/prior_bounds.yaml` under `chi2.epsilon`.

## Output

Write `best_params.json` with your current best estimate — update after every call:

```json
{"om": ..., "ob": ..., "h": ..., "ns": ..., "as_": ..., "w0": ...}
```

See `CLAUDE.md` for procedural rules (journal, best_params.json, stopping criteria).

---

**Begin calibration now.** Read `obs_pk.npy`, write your first journal entry, and make your first simulator call.
