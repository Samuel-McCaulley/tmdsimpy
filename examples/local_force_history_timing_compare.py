#!/usr/bin/env python3
"""
Compare local_force_history runtime for Iwan, Bouc-Wen, and NormStiffness.

The benchmark runs for Nt in [2^4, 2^6, 2^8, 2^10, 2^12], repeats each case
for multiple trials, and reports the minimum runtime for each model/Nt pair.
"""

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_tmdsimpy")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_tmdsimpy")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import tmdsimpy.utils.harmonic as hutils
from tmdsimpy.nlforces.bouc_wen import BoucWenForce
from tmdsimpy.nlforces.normstiffness import NormStiffness
from tmdsimpy.nlforces.vector_iwan4 import VectorIwan4


NT_VALUES = [1 << 4, 1 << 6, 1 << 8, 1 << 10, 1 << 12]
HARMONICS = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int)


def build_models():
    """Construct a representative SDOF model set."""
    q = np.array([[1.0]])
    t = np.array([[1.0]])

    models = {
        "Iwan (VectorIwan4)": VectorIwan4(
            q,
            t,
            kt=1000.0,
            Fs=1.0,
            chi=-0.99,
            beta=0.01,
        ),
        "Bouc-Wen": BoucWenForce(
            q,
            t,
            A=1.0,
            beta=0.6,
            gamma=0.4,
            n=2.0,
        ),
        "NormStiffness": NormStiffness(
            q,
            t,
            kt=1000.0,
            b=4.0,
            s=2e-3,
        ),
    }

    return models


def make_time_series(nt, amplitude, omega):
    """Generate displacement, velocity, and harmonic basis arrays."""
    nhc = hutils.Nhc(HARMONICS)
    unl_harmonics = np.zeros((nhc, 1))
    unl_harmonics[1, 0] = amplitude

    unlt = hutils.time_series_deriv(nt, HARMONICS, unl_harmonics, 0)
    unltdot = omega * hutils.time_series_deriv(nt, HARMONICS, unl_harmonics, 1)
    cst = hutils.time_series_deriv(nt, HARMONICS, np.eye(nhc), 0)
    unlth0 = unl_harmonics[0, :]

    return unlt, unltdot, cst, unlth0


def run_trials(model, unlt, unltdot, cst, unlth0, trials):
    """Run repeated timings and return both trial times and minimum."""
    def eval_local_force_with_gradients():
        ft, dfduh, dfdudh = model.local_force_history(
            unlt,
            unltdot,
            HARMONICS,
            cst,
            unlth0,
        )
        # Touch force and harmonic gradient outputs so gradient formation is
        # explicitly part of the timed path.
        return (
            float(ft[-1, 0])
            + float(dfduh[-1, 0, 0, -1])
            + float(dfdudh[-1, 0, 0, -1])
        )

    _ = eval_local_force_with_gradients()

    trial_times = []
    probes = []
    for _ in range(trials):
        start = time.perf_counter()
        probes.append(eval_local_force_with_gradients())
        trial_times.append(time.perf_counter() - start)

    return trial_times, min(trial_times), probes[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local_force_history runtime for three models."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of timing trials per model/Nt (default: 100).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=2e-3,
        help="Displacement harmonic amplitude used to generate unlt (default: 2e-3).",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Fundamental frequency multiplier for unltdot (default: 1.0).",
    )

    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")

    models = build_models()
    model_names = list(models.keys())

    min_times = {name: [] for name in model_names}
    all_trials = {name: [] for name in model_names}

    print("Benchmark setup")
    print(f"  Nt values: {NT_VALUES}")
    print(f"  Trials per case: {args.trials}")
    print(f"  Harmonics: {HARMONICS.tolist()}")
    print(f"  Amplitude: {args.amplitude}")
    print(f"  Omega: {args.omega}")
    print("  Includes harmonic gradients (dfduh, dfdudh): True")
    print("")

    for nt in NT_VALUES:
        unlt, unltdot, cst, unlth0 = make_time_series(nt, args.amplitude, args.omega)

        for name in model_names:
            trials, min_t, _probe = run_trials(
                models[name],
                unlt,
                unltdot,
                cst,
                unlth0,
                args.trials,
            )
            all_trials[name].append(trials)
            min_times[name].append(min_t)

    header = (
        f"{'Nt':>8}"
        f"{'Iwan min [s]':>18}"
        f"{'Bouc-Wen min [s]':>20}"
        f"{'NormStiffness min [s]':>25}"
    )
    print("Minimum runtime summary")
    print(header)
    print("-" * len(header))

    iwan_name = "Iwan (VectorIwan4)"
    bouc_name = "Bouc-Wen"
    norm_name = "NormStiffness"
    for idx, nt in enumerate(NT_VALUES):
        print(
            f"{nt:8d}"
            f"{min_times[iwan_name][idx]:18.6e}"
            f"{min_times[bouc_name][idx]:20.6e}"
            f"{min_times[norm_name][idx]:25.6e}"
        )

    print("\nPer-case trial times [s]")
    for idx, nt in enumerate(NT_VALUES):
        print(f"\nNt = {nt}")
        for name in model_names:
            trials = ", ".join(f"{x:.6e}" for x in all_trials[name][idx])
            print(f"  {name:20s} min={min_times[name][idx]:.6e}, trials=[{trials}]")


if __name__ == "__main__":
    main()
