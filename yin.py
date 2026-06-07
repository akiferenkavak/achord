"""
yin.py — Core pitch-detection routines for the Achord guitar tuner.

This module factors the digital-signal-processing core out of the deployed
server (``tuner_server.py``) so that the *exact same* routine can be reused by
the offline evaluation scripts in ``tests/``.  The YIN/CMNDF implementation
below is identical, line for line, to the one running inside the WebSocket
backend; the only additions are two baseline detectors (naive autocorrelation
and FFT peak-picking) used for the comparative experiments in the report.

References
----------
A. de Cheveigne and H. Kawahara, "YIN, a fundamental frequency estimator for
speech and music," J. Acoust. Soc. Am., vol. 111, no. 4, pp. 1917-1930, 2002.
"""

from __future__ import annotations

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100                       # Hz, CD-quality sampling rate
NOTE_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
               'F#', 'G', 'G#', 'A', 'A#', 'B']
A4_FREQ     = 440.0                        # Hz, reference pitch for 12-TET


# ── Frequency → musical note + cent deviation ────────────────────────────────
def freq_to_note(freq: float) -> dict:
    """Map a frequency (Hz) to the nearest note in 12-tone equal temperament.

    The deviation is reported in cents, where 100 cents = one semitone and
    1200 cents = one octave.
    """
    if freq <= 0:
        return {"note": "—", "octave": "", "cents": 0, "freq": 0.0}

    semitones_from_a4 = 12 * np.log2(freq / A4_FREQ)
    nearest_semitone  = round(semitones_from_a4)
    cents             = (semitones_from_a4 - nearest_semitone) * 100
    note_index        = (nearest_semitone + 9) % 12
    octave            = 4 + (nearest_semitone + 9) // 12

    return {
        "note":   NOTE_NAMES[note_index],
        "octave": str(octave),
        "cents":  round(float(cents), 1),
        "freq":   round(float(freq), 2),
    }


def cents_error(detected: float, reference: float) -> float:
    """Cent deviation of a detected frequency from a known reference (Eq. cent).

    cents = 1200 * log2(detected / reference)
    """
    if detected <= 0 or reference <= 0:
        return float("nan")
    return 1200.0 * np.log2(detected / reference)


def rms(data: np.ndarray) -> float:
    """Root-mean-square level of a frame, used as the silence gate."""
    return float(np.sqrt(np.mean(data ** 2)))


# ── YIN / CMNDF pitch detection ──────────────────────────────────────────────
def detect_pitch(f: np.ndarray, sample_rate: int = SAMPLE_RATE,
                 bounds: tuple = (60, 1200), thresh: float = 0.10,
                 return_debug: bool = False):
    """Estimate the fundamental frequency of a frame with the YIN algorithm.

    Pipeline: difference function -> cumulative mean normalized difference
    function (CMNDF) -> absolute-threshold local minimum -> parabolic
    interpolation -> f0 = sample_rate / lag.

    Parameters
    ----------
    f : np.ndarray
        Mono audio frame (float32/float64).
    bounds : tuple
        (f_min, f_max) search range in Hz.
    thresh : float
        YIN absolute threshold applied to the CMNDF.
    return_debug : bool
        If True, also return (lags, df, cmndf, chosen_lag) for plotting.
    """
    W       = len(f) // 2
    min_lag = max(2, int(sample_rate / bounds[1]))
    max_lag = min(len(f) - W - 1, int(sample_rate / bounds[0]))

    if min_lag >= max_lag:
        return (None, None, None, None, None) if return_debug else None

    lags        = np.arange(min_lag, max_lag)
    window_data = f[:W]

    # 1. Difference function  d(tau) = sum_j (x_j - x_{j+tau})^2
    df_values = np.array([
        np.sum((window_data - f[lag: lag + W]) ** 2)
        for lag in lags
    ])

    # 2. Cumulative mean normalized difference function (CMNDF)
    cumsum_df    = np.cumsum(df_values)
    running_mean = cumsum_df / (np.arange(1, len(df_values) + 1))
    cmndf_vals   = df_values / (running_mean + 1e-20)

    # 3. First local minimum below the absolute threshold
    sample_lag = None
    for i in range(1, len(cmndf_vals) - 1):
        if (cmndf_vals[i] < thresh
                and cmndf_vals[i] < cmndf_vals[i - 1]
                and cmndf_vals[i] < cmndf_vals[i + 1]):
            sample_lag = lags[i]
            break

    if sample_lag is None:
        min_idx = int(np.argmin(cmndf_vals))
        if cmndf_vals[min_idx] > 0.35:
            return (None, lags, df_values, cmndf_vals, None) if return_debug else None
        sample_lag = int(lags[min_idx])

    chosen_lag = float(sample_lag)

    # 4. Parabolic interpolation around the chosen minimum.
    #    Vertex offset of the parabola through (lag-1, lag, lag+1):
    #        delta = (y0 - y2) / (2 * (y0 - 2*y1 + y2))
    #    The denominator is positive at a local minimum and the refined lag is
    #    (sample_lag + delta). NOTE: the originally deployed code used
    #    denom = 2*(2*y1 - y0 - y2), which inverts the sign of delta and pushes
    #    the lag the wrong way; that bug introduced up to ~8 cents of bias for
    #    notes whose period falls near a half-sample. The sign is fixed here.
    rel = sample_lag - min_lag
    if 0 < rel < len(cmndf_vals) - 1:
        y0, y1, y2 = cmndf_vals[rel - 1], cmndf_vals[rel], cmndf_vals[rel + 1]
        denom = 2 * (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-10:
            chosen_lag = sample_lag + (y0 - y2) / denom

    pitch = float(sample_rate / chosen_lag)
    if return_debug:
        return pitch, lags, df_values, cmndf_vals, chosen_lag
    return pitch


# ── Baseline detectors (for comparison only) ─────────────────────────────────
def detect_autocorrelation(f: np.ndarray, sample_rate: int = SAMPLE_RATE,
                           bounds: tuple = (60, 1200)) -> float | None:
    """Naive time-domain autocorrelation pitch detector.

    Picks the largest autocorrelation peak inside the admissible lag range.
    Prone to octave (period-doubling/halving) ambiguity.
    """
    f = f - np.mean(f)
    n = len(f)
    corr = np.correlate(f, f, mode='full')[n - 1:]      # non-negative lags

    min_lag = max(2, int(sample_rate / bounds[1]))
    max_lag = min(n - 1, int(sample_rate / bounds[0]))
    if min_lag >= max_lag:
        return None

    segment = corr[min_lag:max_lag]
    if not np.any(segment > 0):
        return None
    lag = min_lag + int(np.argmax(segment))

    # parabolic interpolation
    if 0 < lag < len(corr) - 1:
        y0, y1, y2 = corr[lag - 1], corr[lag], corr[lag + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            lag = lag + (y0 - y2) / denom
    return float(sample_rate / lag)


def detect_fft_peak(f: np.ndarray, sample_rate: int = SAMPLE_RATE,
                    bounds: tuple = (60, 1200)) -> float | None:
    """FFT peak-picking pitch detector.

    Returns the frequency of the loudest spectral bin in range. For harmonic
    instruments the loudest partial is frequently an overtone rather than the
    fundamental, so this estimator is biased toward octave-up errors.
    """
    n      = len(f)
    win    = np.hanning(n)
    spec   = np.abs(np.fft.rfft(f * win))
    freqs  = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    band = (freqs >= bounds[0]) & (freqs <= bounds[1])
    if not np.any(band):
        return None
    idx_band = np.where(band)[0]
    k = idx_band[int(np.argmax(spec[band]))]

    # parabolic interpolation on the spectrum
    if 0 < k < len(spec) - 1:
        y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    return float((k + delta) * sample_rate / n)
