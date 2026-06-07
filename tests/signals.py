"""
signals.py — Synthetic guitar-tone generator with known ground-truth pitch.

A plucked string is modelled as a sum of harmonic partials with a decaying
amplitude profile, optional random partial phases, and additive white Gaussian
noise at a prescribed signal-to-noise ratio.  Because the fundamental f0 is
prescribed exactly, the cent error of any detector can be measured against an
exact reference (this is what makes the evaluation in the report quantitative).
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
FRAME_SIZE  = 4096

# Standard guitar tuning (note, fundamental frequency in Hz)
GUITAR_STRINGS = [
    ("E2",  82.41),
    ("A2", 110.00),
    ("D3", 146.83),
    ("G3", 196.00),
    ("B3", 246.94),
    ("E4", 329.63),
]

# A "bright" harmonic profile in which the 2nd partial is the strongest one,
# representative of a string plucked near the bridge. This is the situation in
# which spectral peak-picking is most likely to lock onto an overtone.
BRIGHT_PROFILE = np.array([0.7, 1.0, 0.6, 0.5, 0.35, 0.25, 0.18, 0.12])

# A "mellow" profile with a dominant fundamental (plucked over the sound hole).
MELLOW_PROFILE = np.array([1.0, 0.6, 0.4, 0.28, 0.2, 0.14, 0.1, 0.07])


def guitar_frame(f0: float,
                 snr_db: float | None = 25.0,
                 profile: np.ndarray = BRIGHT_PROFILE,
                 n: int = FRAME_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 level: float = 0.1,
                 decay: float = 1.5,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate one synthetic guitar frame at fundamental ``f0``.

    Parameters
    ----------
    f0 : float
        Fundamental frequency in Hz (ground truth).
    snr_db : float or None
        Signal-to-noise ratio in dB. ``None`` -> no added noise.
    profile : np.ndarray
        Relative amplitudes of the harmonic partials.
    level : float
        Target RMS level of the clean tone (kept above the 0.008 silence gate).
    decay : float
        Exponential amplitude decay across the frame (per frame length).
    rng : np.random.Generator
        Optional seeded generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n) / sample_rate
    sig = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate(profile, start=1):
        if k * f0 >= sample_rate / 2:          # skip partials above Nyquist
            break
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * k * f0 * t + phase)

    # mild exponential decay envelope (a plucked note loses energy over time)
    sig *= np.exp(-decay * t / (n / sample_rate))

    # normalise to the requested RMS level
    cur = np.sqrt(np.mean(sig ** 2))
    if cur > 0:
        sig *= level / cur

    # additive white Gaussian noise at the requested SNR
    if snr_db is not None:
        sig_power   = np.mean(sig ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        sig += rng.normal(0.0, np.sqrt(noise_power), n)

    return sig.astype(np.float32)
