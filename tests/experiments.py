"""
experiments.py — Reproducible evaluation of the Achord pitch detector.

Running this script regenerates every quantitative result and every figure used
in the final report:

    python tests/experiments.py

Outputs
-------
results/results.json          machine-readable summary of all experiments
results/*.csv                 per-experiment tables
report/figures/*.pdf          vector figures for the LaTeX report

All randomness is seeded, so the numbers are reproducible.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# make the project root importable when run from anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yin import (detect_pitch, detect_autocorrelation, detect_fft_peak,
                 cents_error, SAMPLE_RATE)
from tests.signals import (guitar_frame, GUITAR_STRINGS,
                           BRIGHT_PROFILE, MELLOW_PROFILE, FRAME_SIZE)

RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR     = os.path.join(ROOT, "report", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SEED       = 2026
N_TRIALS   = 200            # Monte-Carlo trials per condition
OCTAVE_TOL = 600.0          # |cents| beyond this counts as an octave-class error
NOTE_TOL   = 50.0           # |cents| within this counts as the correct note

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})

DETECTORS = {
    "YIN/CMNDF":      detect_pitch,
    "Autocorrelation": detect_autocorrelation,
    "FFT peak":        detect_fft_peak,
}


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


# ── Experiment 1: accuracy per string (YIN, near-clean) ──────────────────────
def exp_accuracy(snr_db=25.0):
    rng = np.random.default_rng(SEED)
    rows, summary = [], []
    for note, f0 in GUITAR_STRINGS:
        errs = []
        single_measured = None
        for t in range(N_TRIALS):
            frame = guitar_frame(f0, snr_db=snr_db, profile=BRIGHT_PROFILE, rng=rng)
            est = detect_pitch(frame)
            if est is None:
                continue
            if single_measured is None:
                single_measured = est
            errs.append(cents_error(est, f0))
        errs = np.array(errs)
        mean_meas = f0 * 2 ** (np.mean(errs) / 1200.0)
        rows.append([note, f"{f0:.2f}", f"{mean_meas:.2f}",
                     f"{np.mean(errs):+.2f}", f"{np.std(errs):.2f}",
                     f"{np.max(np.abs(errs)):.2f}"])
        summary.append({
            "note": note, "target": f0,
            "mean_measured": float(mean_meas),
            "mean_cents": float(np.mean(errs)),
            "std_cents": float(np.std(errs)),
            "max_abs_cents": float(np.max(np.abs(errs))),
            "n": int(len(errs)),
        })
    _write_csv(os.path.join(RESULTS_DIR, "accuracy.csv"),
               ["Note", "Target (Hz)", "Mean measured (Hz)",
                "Mean cents", "Std cents", "Max |cents|"], rows)
    return {"snr_db": snr_db, "n_trials": N_TRIALS, "per_string": summary}


# ── Experiment 2: method comparison across SNR ───────────────────────────────
def exp_methods(snr_list=(-5, 0, 5, 10, 15, 20, 25, 30)):
    rng = np.random.default_rng(SEED + 1)
    # aggregate over all strings and both timbres
    profiles = [BRIGHT_PROFILE, MELLOW_PROFILE]
    per_method = {name: {"snr": [], "correct": [], "octave": [], "mae": []}
                  for name in DETECTORS}

    for snr in snr_list:
        for name, fn in DETECTORS.items():
            correct = octave = total = 0
            abs_errs = []
            for prof in profiles:
                for note, f0 in GUITAR_STRINGS:
                    for _ in range(N_TRIALS):
                        frame = guitar_frame(f0, snr_db=snr, profile=prof, rng=rng)
                        est = fn(frame)
                        total += 1
                        if est is None or est <= 0:
                            continue
                        ce = cents_error(est, f0)
                        abs_errs.append(abs(ce))
                        if abs(ce) <= NOTE_TOL:
                            correct += 1
                        elif abs(ce) >= OCTAVE_TOL:
                            octave += 1
            per_method[name]["snr"].append(snr)
            per_method[name]["correct"].append(100.0 * correct / total)
            per_method[name]["octave"].append(100.0 * octave / total)
            per_method[name]["mae"].append(float(np.median(abs_errs)) if abs_errs else float("nan"))

    # summary table at a representative moderate SNR (10 dB)
    idx = list(snr_list).index(10) if 10 in snr_list else len(snr_list) // 2
    rows, summary = [], {}
    for name in DETECTORS:
        c = per_method[name]["correct"][idx]
        o = per_method[name]["octave"][idx]
        m = per_method[name]["mae"][idx]
        rows.append([name, f"{c:.1f}", f"{o:.1f}", f"{m:.2f}"])
        summary[name] = {"correct_pct": c, "octave_pct": o, "median_abs_cents": m}
    _write_csv(os.path.join(RESULTS_DIR, "methods.csv"),
               ["Method", "Correct-note %", "Octave-error %",
                "Median |cents|"], rows)
    return {"snr_used_for_table": snr_list[idx], "curves": per_method,
            "table": summary, "snr_list": list(snr_list)}


# ── Experiment 3: YIN robustness vs SNR ──────────────────────────────────────
def exp_noise(snr_list=(-10, -5, 0, 5, 10, 15, 20, 25, 30)):
    rng = np.random.default_rng(SEED + 2)
    snrs, correct, mae = [], [], []
    for snr in snr_list:
        ok = total = 0
        errs = []
        for note, f0 in GUITAR_STRINGS:
            for _ in range(N_TRIALS):
                frame = guitar_frame(f0, snr_db=snr, profile=BRIGHT_PROFILE, rng=rng)
                est = detect_pitch(frame)
                total += 1
                if est is None:
                    continue
                ce = cents_error(est, f0)
                if abs(ce) <= NOTE_TOL:
                    ok += 1
                    errs.append(abs(ce))
        snrs.append(snr)
        correct.append(100.0 * ok / total)
        mae.append(float(np.mean(errs)) if errs else float("nan"))
    return {"snr": snrs, "correct_pct": correct, "mean_abs_cents_correct": mae}


# ── Experiment 4: per-frame compute latency ──────────────────────────────────
def exp_latency(runs=1000):
    rng = np.random.default_rng(SEED + 3)
    frames = [guitar_frame(GUITAR_STRINGS[i % 6][1], snr_db=20, rng=rng)
              for i in range(runs)]
    # warm-up
    for fr in frames[:20]:
        detect_pitch(fr)
    times = []
    for fr in frames:
        t0 = time.perf_counter()
        detect_pitch(fr)
        times.append((time.perf_counter() - t0) * 1000.0)   # ms
    times = np.array(times)
    frame_ms = 1000.0 * FRAME_SIZE / SAMPLE_RATE
    return {
        "compute_mean_ms": float(np.mean(times)),
        "compute_median_ms": float(np.median(times)),
        "compute_std_ms": float(np.std(times)),
        "compute_p95_ms": float(np.percentile(times, 95)),
        "frame_acquisition_ms": float(frame_ms),
        "runs": runs,
    }


# ── Figures ──────────────────────────────────────────────────────────────────
def fig_waveform_and_cmndf():
    rng = np.random.default_rng(SEED + 10)
    f0 = 82.41                                   # E2
    frame = guitar_frame(f0, snr_db=20, profile=BRIGHT_PROFILE, rng=rng)
    pitch, lags, df, cmndf, chosen = detect_pitch(frame, return_debug=True)

    # waveform
    t_ms = np.arange(len(frame)) / SAMPLE_RATE * 1000.0
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(t_ms[:1500], frame[:1500], lw=0.8, color="#1f77b4")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Synthetic E2 frame (f0 = {f0} Hz, SNR = 20 dB)")
    fig.savefig(os.path.join(FIG_DIR, "waveform.pdf"))
    plt.close(fig)

    # difference function + CMNDF
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 4.4), sharex=True)
    a1.plot(lags, df, color="#9467bd", lw=1.0)
    a1.set_ylabel("d(τ)")
    a1.set_title("Difference function and CMNDF (E2 frame)")
    a2.plot(lags, cmndf, color="#2ca02c", lw=1.0, label="CMNDF d'(τ)")
    a2.axhline(0.10, color="#d62728", ls="--", lw=1.0, label="threshold = 0.10")
    if chosen is not None:
        a2.axvline(chosen, color="#ff7f0e", ls=":", lw=1.2,
                   label=f"chosen lag → {pitch:.1f} Hz")
    a2.set_xlabel("Lag τ (samples)")
    a2.set_ylabel("d'(τ)")
    a2.set_ylim(0, 1.4)
    a2.legend(fontsize=8, loc="upper right")
    fig.savefig(os.path.join(FIG_DIR, "cmndf.pdf"))
    plt.close(fig)
    return {"e2_detected": float(pitch), "e2_chosen_lag": float(chosen)}


def fig_cent_error(acc):
    notes = [s["note"] for s in acc["per_string"]]
    means = [s["mean_cents"] for s in acc["per_string"]]
    stds  = [s["std_cents"] for s in acc["per_string"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(notes, means, yerr=stds, capsize=4, color="#2ca02c", alpha=0.85)
    ax.axhline(5, color="#d62728", ls="--", lw=1.0, label="±5 cent target")
    ax.axhline(-5, color="#d62728", ls="--", lw=1.0)
    ax.set_ylabel("Cent error")
    ax.set_xlabel("Guitar string")
    ax.set_title(f"YIN cent error per string "
                 f"(SNR = {acc['snr_db']:.0f} dB, {acc['n_trials']} trials)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, "cent_error.pdf"))
    plt.close(fig)


def fig_methods(methods):
    curves = methods["curves"]
    colors = {"YIN/CMNDF": "#2ca02c", "Autocorrelation": "#1f77b4", "FFT peak": "#d62728"}
    # correct-note rate vs SNR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for name, c in curves.items():
        a1.plot(c["snr"], c["correct"], "o-", color=colors[name], label=name, ms=4)
        a2.plot(c["snr"], c["octave"], "o-", color=colors[name], label=name, ms=4)
    a1.set_xlabel("SNR (dB)"); a1.set_ylabel("Correct-note rate (%)")
    a1.set_title("Detection accuracy"); a1.set_ylim(0, 105)
    a2.set_xlabel("SNR (dB)"); a2.set_ylabel("Octave-error rate (%)")
    a2.set_title("Octave-class errors")
    a1.legend(fontsize=8, loc="lower right")
    fig.savefig(os.path.join(FIG_DIR, "methods.pdf"))
    plt.close(fig)


def fig_noise(noise):
    fig, ax1 = plt.subplots(figsize=(6.2, 3.0))
    ax1.plot(noise["snr"], noise["correct_pct"], "o-", color="#2ca02c",
             label="Correct-note rate")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Correct-note rate (%)", color="#2ca02c")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(noise["snr"], noise["mean_abs_cents_correct"], "s--", color="#d62728",
             label="Mean |cent| error")
    ax2.set_ylabel("Mean |cent| error (correct)", color="#d62728")
    ax1.set_title("YIN robustness to additive noise")
    fig.savefig(os.path.join(FIG_DIR, "noise.pdf"))
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Running experiments (seed=%d, %d trials/condition)…" % (SEED, N_TRIALS))
    acc     = exp_accuracy()
    print("  [1/4] accuracy done")
    methods = exp_methods()
    print("  [2/4] method comparison done")
    noise   = exp_noise()
    print("  [3/4] noise sweep done")
    latency = exp_latency()
    print("  [4/4] latency done")

    dbg = fig_waveform_and_cmndf()
    fig_cent_error(acc)
    fig_methods(methods)
    fig_noise(noise)
    print("  figures written to", FIG_DIR)

    results = {
        "config": {"seed": SEED, "n_trials": N_TRIALS, "sample_rate": SAMPLE_RATE,
                   "frame_size": FRAME_SIZE, "octave_tol_cents": OCTAVE_TOL,
                   "note_tol_cents": NOTE_TOL},
        "accuracy": acc,
        "methods": methods,
        "noise": noise,
        "latency": latency,
        "debug": dbg,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("  results written to", os.path.join(RESULTS_DIR, "results.json"))

    # console summary
    print("\n=== Accuracy (per string) ===")
    for s in acc["per_string"]:
        print(f"  {s['note']:>3}  target {s['target']:7.2f} Hz  "
              f"mean {s['mean_cents']:+.2f} ± {s['std_cents']:.2f} cents  "
              f"max |{s['max_abs_cents']:.2f}|")
    print("\n=== Method comparison @ %d dB SNR ===" % methods["snr_used_for_table"])
    for name, v in methods["table"].items():
        print(f"  {name:>16}  correct {v['correct_pct']:5.1f}%  "
              f"octave {v['octave_pct']:5.1f}%  median |{v['median_abs_cents']:.2f}| cents")
    print("\n=== Latency ===")
    print(f"  compute mean {latency['compute_mean_ms']:.3f} ms, "
          f"p95 {latency['compute_p95_ms']:.3f} ms, "
          f"frame acquisition {latency['frame_acquisition_ms']:.1f} ms")


if __name__ == "__main__":
    main()
