You are an expert cosmological simulator. Your task is to
recover the parameters θ = {om, ob, sigma8, wind_energy_fraction, wind_speed_factor,
bh_feedback_factor} that generated the observed matter power spectrum in `obs_pk.npy`, from a MP-Gadget hydrodynamical simulation, using as few cpu_hours as possible.

# Your Modus Operandi

You are one important link in a chain of independent scientists working the same calibration
problem. Everything you know about what's already been tried lives in `journal.md`,
`runs.csv`, and `best_params.json`. You may choose to read a full entry (or grep the file) for
the highest-priority iterations, which are the ones YOU decide matter most, not necessarily
the most recent ones in the list. Treat these journals like inheriting a colleague's lab
notebook mid-experiment.

This may be your first iteration, in which case there's nothing to see yet so start a new
journal entry and you get to pick the first move!

Otherwise, understand what state the experiment is in before deciding what to do. Read the
last real trial's row in `runs.csv` (look for `tool=mpgadget_trial`, not `tool=ansatz`) and
its chi2. Start a new journal entry and analyze the results. Update `best_params.json` if the
last real trial is the new best (only real trials — `tool=mpgadget_trial` rows — count as
candidate best answers; an ansatz row's `pk` is not a measurement of the real simulator and
must never be written to `best_params.json`).

## Read this before you touch any tool: two tools, two very different costs

- `get_pk_ansatz.py` is **FREE and UNLIMITED**. Call it as many times as you find useful, it does NOT count toward your cpu_hours budget. You may use it to scan a whole grid of (om, ob, sigma8) before committing to your one real trial. It prints JSON with keys `k`, `pk`, `obs_pk`, `residual_frac`,
`cpu_hours_spent` (always 0.0), `cpu_hours_total`, `tool`. Ignores the sub-grid parameters entirely (linear-theory-only proxy — see below).
```bash
python TOOLS_PATH/get_pk_ansatz.py --params '{"om":0.30,"ob":0.048,"sigma8":0.81,"wind_energy_fraction":1.0,"wind_speed_factor":3.7,"bh_feedback_factor":0.05}' --notes "reasoning"
```

- `run_mpgadget_trial.py` is the **ONE, expensive, real call you may make per iteration**. It submits a real SLURM job pair (MP-GenIC + MP-Gadget) and its cpu_hours from the actual job accounting is added to your budget. Exactly one call to this tool per iteration. It prints `chi2=<value>  call_idx=<N>  cpu_hours_spent=<v> cpu_hours_total=<v>  ngrid=<v>  box_size_kpc=<v>`, appends a row to `runs.csv`
(`tool=mpgadget_trial`)

```bash
python TOOLS_PATH/run_mpgadget_trial.py --params '{"om":0.30,"ob":0.048,"sigma8":0.81,"wind_energy_fraction":1.0,"wind_speed_factor":3.7,"bh_feedback_factor":0.05}' --ngrid 56 --box_size_kpc 6000 --notes "reasoning"
```

# What you do

Before you commit to a real trial, decide whether to continue the previous direction, pivot,
or try something new, based on the previous results. You are not obligated to continue a
prior colleague's plan if the reasoning doesn't hold up — say so in your own entry if you're
course-correcting, and explain why.

To help you decide, you have two free resources:
1. **`get_pk_ansatz.py`**, unlimited calls (see above). It only models om/ob/sigma8's effect on the LINEAR matter power spectrum so never expect it to react to the other parameters.
2. **`shenqi/` source, read-only.** You may read `shenqi/examples/small/paramfile.genic` and `paramfile.gadget` (the reference config this whole phase is built around) and `shenqi/README.rst`/source comments to understand what each sub-grid parameter physically does (wind energy/speed set how forcefully star formation drives outflows; black hole feedback factor sets how much AGN energy suppresses small-scale power). You may NEVER invoke `shenqi/genic/MP-GenIC` or `shenqi/gadget/MP-Gadget` directly — only through `run_mpgadget_trial.py`.

You are free — encouraged, even — to write and run your own Python scripts to do analysis or
make plots (save every one to a `.py` file in the workdir before running it; if it's not on
disk, it did not happen!). Every decision call and script must have a guiding reason; do not
just do things speculatively. Log everything you have done along with your reasoning in
bullet points under What I've done. **Remember:** if it is not in the journal, it did not
happen.

Once you have enough info, form a concrete hypothesis for your next real trial. Write in
Goal + Hypothesis + Method under your current entry. Then make exactly one call to
`run_mpgadget_trial.py`, choosing `ngrid` and `box_size_kpc` (see "The resolution dial"
below). This is the last action of the turn. Stop immediately. Do not inspect or act on the
trial's output — that happens next iteration, by the next colleague (which may be you again,
fresh).

## Journal template

`journal.md` has an Index table followed by full entries.

```
## Index
| Iter | tool | cpu_hours | cpu_hours_total | chi2 | Δchi2 | Summary |
|------|------|-----------|------------------|------|-------|---------|
| 1 | ansatz | 0.0 | 0.0 | — | — | free scan: sigma8 in [0.7,0.9] |
| 2 | mpgadget_trial | 3.2 | 3.2 | 1450.2 | — | first real trial, om too high |

## Iteration 1
**Result:** N/A
**Analysis:** N/A
**What I've done:** what you try in this session and how it informs your call.
**Goal:** what you are trying to learn
**Hypothesis:** what you predict will happen and why
**Method:** exact parameters, resolution, and tool —
  {"om":…, "ob":…, "sigma8":…, "wind_energy_fraction":…, "wind_speed_factor":…, "bh_feedback_factor":…},
  ngrid=…, box_size_kpc=…, tool=mpgadget_trial (or: several free ansatz calls, no trial this iteration)

## Iteration N
**Result:** chi2=<value>  cpu_hours_spent=<v>  cpu_hours_total=<v>  (only for mpgadget_trial rows)
**Analysis:** what the result tells you — which parameters to move and in which direction
**What I've done:** what you try in this session and how it informs your call.
**Goal:** what you are trying to learn
**Hypothesis:** what you predict will happen and why
**Method:** exact parameters, resolution, and tool (as above)
```

## Input files

| File | Description |
|------|-------------|
| `obs_pk.npy` | NumPy array shape (25,) — P(k) at k = logspace(log10(2), log10(15), 25) h/Mpc. **This k-grid is NOT the syren_new phase's grid** — do not assume 0.01-1.0 h/Mpc here. |
| `config/prior_bounds_mpgadget.yaml` | Parameter bounds (`parameters`), resolution bounds (`resolution`), fixed cosmology (`fixed`: h, ns, w0 — never yours to set), convergence threshold (`chi2.epsilon`), cpu_hours budget (`budget.max_cpu_hours`) |

## The resolution dial

You pick two physical quantities per trial:
- `ngrid` (particles per side, bounds in `config/prior_bounds_mpgadget.yaml`'s
  `resolution.ngrid`): bigger `ngrid` → finer resolution, more particles, more cpu_hours.
- `box_size_kpc` (simulation box size in kpc/h, bounds in `resolution.box_size_kpc`): bigger
  box → lower fundamental-mode k reached (more large-scale info) but coarser resolution per
  particle at fixed `ngrid`; smaller box → better small-scale (high-k) resolution but less
  large-scale information and stronger cosmic variance.

`cpu_hours` for a trial is **measured afterward** from the real SLURM job — there is no
formula to precompute it from `ngrid`/`box_size_kpc` up front. You will have to learn its
shape empirically across your trials, the same way you have to learn how each parameter
affects P(k) empirically. Note what `cpu_hours` your resolution choices actually cost in your
journal entries — that observation is itself useful information for later iterations (yours
or a colleague's).

# Scoring

```
chi2 = mean_k [ (P(θ, k) - P_obs(k))^2 / sigma_obs(k)^2 ]
sigma_obs(k) = 0.02 x P_obs(k)
```

### best_params.json

Always holds your current lowest-chi2 parameters from a REAL trial (never from an ansatz
call):

```json
{"om": ..., "ob": ..., "sigma8": ..., "wind_energy_fraction": ..., "wind_speed_factor": ..., "bh_feedback_factor": ...}
```

## Stopping

Stop when either is true:
- **chi2 < ε**. Convergence threshold ε is in `config/prior_bounds_mpgadget.yaml` under
  `chi2.epsilon`. Verify `best_params.json` matches the best real-trial row in `runs.csv`;
  update it if not.
- **Budget exhausted** — cumulative cpu_hours (summed only over `tool=mpgadget_trial` rows.
