# Cosmological Parameter Calibration

Calibrate a cosmological simulation to match a target matter power spectrum P(k).

## Your task

Find cosmological parameters θ = {om, ob, h, ns, as_, w0} that reproduce the observed P(k) stored in `obs_pk.npy`. Your goal is to reach a good calibration using as few simulator calls as possible — stop as soon as you are confident in your answer.

## Input files

- `obs_pk.npy` — NumPy array, shape (50,) — the observed P(k) at k = logspace(−2, 0, 50) h/Mpc
- `config/prior_bounds.yaml` — parameter bounds and chi2 threshold ε

## Tools

```bash
python TOOLS_PATH/compute_chi2.py --params '{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}' --notes "reasoning"
```
→ prints `chi2=<value>  call_idx=<N>`, appends a row to `runs.csv`.

```bash
python TOOLS_PATH/get_pk.py --params '{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}'
```
→ prints JSON with keys `k`, `pk`, `obs_pk`, `residual_frac`. Also appends to `runs.csv`.

Both tools enforce prior bounds and will error on out-of-range parameters.

You may also write and execute arbitrary Python for analysis.

## What to produce

Write `best_params.json` with your best parameter estimate — update it after every iteration:

```json
{"om": ..., "ob": ..., "h": ..., "ns": ..., "as_": ..., "w0": ...}
```

## Scoring

Your `best_params.json` is evaluated as:

```
chi2 = sum_k [ (P(θ, k) - P_obs(k))^2 / σ(k)^2 ] / N_k
```

where σ(k) = 0.02 × P_obs(k). Pass threshold is chi2 < ε (see `config/prior_bounds.yaml`, key `chi2.epsilon`).

## Notes

The WMAP9 cosmology (fiducial_wmap9 in config) is a physically reasonable starting point near the center of the prior — but the true parameters may differ significantly from it.

See CLAUDE.md for operating instructions.
