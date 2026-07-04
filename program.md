# AgenticSimulator: Cosmological Parameter Calibration

You are calibrating a cosmological simulation to match an observed matter power spectrum P(k).

## Your Goal

Find cosmological parameters θ = {om, ob, h, ns, as_, w0} such that:

    chi2 = sum_k [ (P(θ, k) - P_obs(k))^2 / σ(k)^2 ] / N_k < ε

ε and the maximum call budget are in `config/prior_bounds.yaml` (keys: `chi2.epsilon`, `budget.max_calls`).

## Your Tools

Run from this directory:

```bash
python /path/to/tools/compute_chi2.py --params '{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}' --notes "your reasoning"
```
→ prints `chi2=<value>  call_idx=<N>` and appends a row to `runs.csv`.

```bash
python /path/to/tools/get_pk.py --params '{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}'
```
→ prints JSON with keys `k`, `pk`, `obs_pk`, `residual_frac` for shape analysis. Counts toward your call budget (appends a row to `runs.csv` with chi2 left blank).

You may write and run Python for lightweight analysis (e.g., reading `runs.csv`, computing statistics, fitting residual slopes). You must NOT call `pnl_new_emulated` or any other simulator directly from your own scripts — all simulator evaluations must go through `compute_chi2.py` or `get_pk.py` so they are budget-tracked.

## Your Memory Files

- `runs.csv` — full parameter history with chi2 and call index. **Re-read this at every iteration start.**
- `journal.md` — your freeform reasoning log. **Write to this after every call.**

## Iteration Protocol

### On every iteration:

**Phase 1 — Review**
1. Read `runs.csv` (all rows, or tail if long). Identify: best chi2 so far, which parameters improved it, which had no effect.
2. Read the last 3–5 entries in `journal.md`.
3. Check call count against `budget.max_calls`. If at limit, write final summary and stop.
4. Check if best chi2 < ε. If yes, write final summary and stop.

**Phase 2 — Plateau detection**
If the last 5 chi2 values show no net improvement (last ≤ first), switch strategy:
- Try a direction you have not yet explored
- Try a larger step size
- Try holding all parameters except one fixed and scanning it

**Phase 3 — Physical reasoning and parameter proposal**
Based on the P(k) residual shape, reason about which parameter is most likely wrong:
- Residual positive at all k → overall amplitude too low → increase `as_`
- Residual tilted (positive at low-k, negative at high-k) → spectral index too low → increase `ns`
- Residual shifts the peak position → `om` or `h` (matter-radiation equality scale)
- Residual at small scales only → `ob` (baryon acoustic features)
- `w0` affects growth rate; residual scaling uniformly with k may indicate wrong `w0`

Propose ONE parameter change per iteration. State your reasoning in 2–3 sentences before calling the tool.

**Phase 4 — Execute**
Call `compute_chi2.py` with your proposed parameters.

**Phase 5 — Assess and log**
Compare new chi2 to previous best. Write to `journal.md`:

```
## Iteration N (call_idx=M, chi2=X.XX)
**Changed:** om: 0.281 → 0.290 (trying higher matter density to shift BAO scale)
**Result:** chi2 improved from 145.3 to 121.7
**Residual:** still positive at k < 0.1, suggesting amplitude (as_) is next
**Next:** hold om fixed, increase as_ by 5%
```

## Starting Point

First call: use the WMAP9 fiducial values from `config/prior_bounds.yaml` (`fiducial_wmap9` key) as your initial guess. Note: these are a physically motivated starting point, NOT the hidden answer — theta_fid is drawn uniformly from the prior and may differ significantly.

## Budget Note

Both `compute_chi2.py` and `get_pk.py` count toward the call budget and append rows to `runs.csv`. Use `get_pk.py` for shape diagnostics (it logs the call but leaves chi2 blank); use `compute_chi2.py` when you want the chi2 value recorded.

## Context Compaction Recovery

If your context is reset (compaction event), do not restart from WMAP9. Instead:
1. Read `runs.csv` — the last row's `call_idx` is your current position in the budget
2. Read the last 5 entries in `journal.md` to recover your reasoning state
3. Continue from where you left off

## Stopping

Stop when either:
1. chi2 < ε (success — write "CALIBRATION COMPLETE" to journal.md and print final θ)
2. call_idx = budget.max_calls (budget exhausted — write best θ and chi2 to journal.md)
3. You are clearly stuck in a plateau with no physical reasoning to guide further improvement

On stop, print:
```
FINAL: theta={"om":...} chi2=<value> calls=<N>
```
