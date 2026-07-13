from pathlib import Path

import numpy as np

from simulator.mpgadget_wrapper import MPGadgetSimulator

PROJECT_ROOT = Path(__file__).parent.parent
sim = MPGadgetSimulator(
    shenqi_root=PROJECT_ROOT / "shenqi",
    csv_path=PROJECT_ROOT / "results" / "mpgadget_validation" / "runs.csv",
    partition="debug",  # faster queue turnaround for a one-off validation run
)
pk, cpu_hours = sim(
    params={"om": 0.2814, "ob": 0.0464, "sigma8": 0.81,
            "wind_energy_fraction": 1.0, "wind_speed_factor": 3.7, "bh_feedback_factor": 0.05},
    ngrid=32, box_size_kpc=4000,
    workdir=PROJECT_ROOT / "results" / "mpgadget_validation" / "trial_0",
)
print(f"pk={pk}")
print(f"cpu_hours={cpu_hours:.4f}")
assert pk.shape == (25,)
assert np.all(np.isfinite(pk)) and np.all(pk > 0)
print("VALIDATION PASSED")
