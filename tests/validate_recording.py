"""
validate_recording.py — Validate the YIN detector on *real* guitar audio.

This complements the synthetic benchmark (experiments.py) by running the exact
same detection routine on real recordings, so the tuner can be checked against
an actual instrument.

Two ways to use it
------------------
1) From audio files (WAV/FLAC/OGG work out of the box; MP3/M4A need ffmpeg):

       python tests/validate_recording.py recordings/E2.wav recordings/A2.wav
       python tests/validate_recording.py recordings/*.wav

   If a file name contains a note (e.g. "E2", "A2", "D3", "G3", "B3", "E4"),
   the cent error is measured against that string's exact target frequency.
   A single mixed recording also works; it is segmented and the dominant note
   of each sustained segment is reported.

2) Record straight from the microphone (one string at a time):

       python tests/validate_recording.py --record E2 A2 D3 G3 B3 E4
       python tests/validate_recording.py --record E2 --seconds 3

   Pluck the named string when prompted; each take is saved under recordings/.

Outputs
-------
results/recording_validation.csv      target vs. measured per file/segment
report/figures/recording_<name>.pdf   detected-pitch-vs-time plot per file
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yin import detect_pitch, freq_to_note, cents_error, rms, SAMPLE_RATE

FRAME      = 4096
HOP        = 1024          # overlap for a smoother pitch track
RMS_GATE   = 0.008
REC_DIR    = os.path.join(ROOT, "recordings")
RESULT_DIR = os.path.join(ROOT, "results")
FIG_DIR    = os.path.join(ROOT, "report", "figures")

# Standard tuning targets, used when a file/segment is labelled with a note.
TARGETS = {"E2": 82.41, "A2": 110.00, "D3": 146.83,
           "G3": 196.00, "B3": 246.94, "E4": 329.63}


# ── Audio loading ────────────────────────────────────────────────────────────
def load_audio(path: str) -> np.ndarray:
    """Load an audio file as mono float32 at SAMPLE_RATE."""
    try:
        import librosa
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return y.astype(np.float32)
    except Exception:
        # fallback: WAV via scipy
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        sr, data = wavfile.read(path)
        data = data.astype(np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(np.dtype(data.dtype), np.integer):
            data /= np.iinfo(np.int16).max
        if sr != SAMPLE_RATE:
            data = resample_poly(data, SAMPLE_RATE, sr)
        return data.astype(np.float32)


def record_from_mic(seconds: float, label: str) -> np.ndarray:
    """Record a mono take from the default microphone."""
    import sounddevice as sd
    input(f"  -> Get ready to pluck the {label} string, then press Enter...")
    print(f"  recording {seconds:.0f} s...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    print("  done.")
    return audio.flatten()


# ── Pitch track ──────────────────────────────────────────────────────────────
def pitch_track(audio: np.ndarray):
    """Return (times, freqs, notes, cents) for voiced frames."""
    times, freqs, notes, cents = [], [], [], []
    for start in range(0, len(audio) - FRAME, HOP):
        frame = audio[start:start + FRAME]
        if rms(frame) < RMS_GATE:
            continue
        f = detect_pitch(frame)
        if f is None or not (60 < f < 1200):
            continue
        info = freq_to_note(f)
        times.append(start / SAMPLE_RATE)
        freqs.append(f)
        notes.append(info["note"] + info["octave"])
        cents.append(info["cents"])
    return (np.array(times), np.array(freqs), notes, np.array(cents))


def label_from_name(name: str) -> str | None:
    m = re.search(r"(E2|A2|D3|G3|B3|E4)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def analyse(path_or_label: str, audio: np.ndarray, rows: list):
    name = os.path.splitext(os.path.basename(path_or_label))[0]
    label = label_from_name(name)
    times, freqs, notes, cents = pitch_track(audio)

    if len(freqs) == 0:
        print(f"[{name}] no voiced frames detected (too quiet?)")
        return

    # dominant detected note over the take
    vals, counts = np.unique(notes, return_counts=True)
    dom = vals[int(np.argmax(counts))]
    stable = freqs[np.array(notes) == dom]
    med_f = float(np.median(stable))
    med_note = freq_to_note(med_f)

    target = TARGETS.get(label)
    if target is not None:
        ce = cents_error(med_f, target)
        verdict = f"target {label} {target:.2f} Hz  |  cent error {ce:+.2f}"
    else:
        ce = None
        verdict = "no target label (filename has no note)"

    print(f"[{name}] dominant {dom}  median {med_f:.2f} Hz "
          f"({med_note['cents']:+.1f} cents to nearest note)  |  {verdict}")

    rows.append([name, label or "", f"{target:.2f}" if target else "",
                 dom, f"{med_f:.2f}",
                 f"{ce:+.2f}" if ce is not None else "",
                 f"{med_note['cents']:+.1f}", len(freqs)])

    # pitch-vs-time figure
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(times, freqs, ".", ms=3, color="#1f77b4")
    if target is not None:
        ax.axhline(target, color="#2ca02c", ls="--", lw=1.0,
                   label=f"{label} target {target:.2f} Hz")
        ax.legend(fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Detected frequency (Hz)")
    ax.set_title(f"Detected pitch over time — {name}")
    ax.grid(alpha=0.3)
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"recording_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"        figure -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="audio files to analyse")
    ap.add_argument("--record", nargs="+", metavar="NOTE",
                    help="record these strings from the microphone, e.g. E2 A2")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="recording length per string (default 3 s)")
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = []

    if args.record:
        os.makedirs(REC_DIR, exist_ok=True)
        import soundfile as sf
        for note in args.record:
            audio = record_from_mic(args.seconds, note)
            wav = os.path.join(REC_DIR, f"{note}.wav")
            sf.write(wav, audio, SAMPLE_RATE)
            print(f"  saved {wav}")
            analyse(wav, audio, rows)

    for path in args.files:
        if not os.path.isfile(path):
            print(f"!! not found: {path}")
            continue
        analyse(path, load_audio(path), rows)

    if not rows:
        print("\nNothing analysed. Provide audio files or use --record.")
        print("Example: python tests/validate_recording.py recordings/E2.wav")
        return

    csv_path = os.path.join(RESULT_DIR, "recording_validation.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["File", "Label", "Target (Hz)", "Dominant note",
                    "Median freq (Hz)", "Cent error vs target",
                    "Cents to nearest note", "Voiced frames"])
        w.writerows(rows)
    print(f"\nSummary table -> {csv_path}")


if __name__ == "__main__":
    main()
