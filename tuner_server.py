"""
Gitar Akort Uygulaması - Python Backend (Sunucu Uyumlu)
Gereksinimler: pip install numpy websockets

Mikrofon tarayıcıda açılır → ham Float32 ses verisi WebSocket ile buraya gelir
→ Python CMNDF ile pitch hesaplar → sonuç JSON olarak tarayıcıya geri gönderilir.
PyAudio / PortAudio KULLANILMAZ — bulut sunucularda çalışır.

Çalıştırma:
    python tuner_server.py

Render / sunucu için ortam değişkenleri:
    WS_HOST=0.0.0.0   (varsayılan: localhost)
    WS_PORT=8765       (varsayılan: 8765)
"""

import asyncio
import json
import os
import struct
import numpy as np
import websockets

# ─── Ayarlar ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
WS_HOST     = os.environ.get("WS_HOST", "0.0.0.0")
WS_PORT     = int(os.environ.get("WS_PORT", 8765))

# ─── Kromatik nota isimleri ───────────────────────────────────────────────────
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


# ─── Frekans → Nota + Cent ────────────────────────────────────────────────────
def freq_to_note(freq: float) -> dict:
    if freq <= 0:
        return {"note": "—", "octave": "", "cents": 0, "freq": 0.0}

    semitones_from_a4 = 12 * np.log2(freq / 440.0)
    nearest_semitone  = round(semitones_from_a4)
    cents             = (semitones_from_a4 - nearest_semitone) * 100

    note_index = (nearest_semitone + 9) % 12
    octave     = 4 + (nearest_semitone + 9) // 12
    note_name  = NOTE_NAMES[note_index]

    return {
        "note":   note_name,
        "octave": str(octave),
        "cents":  round(float(cents), 1),
        "freq":   round(float(freq), 2),
    }


# ─── CMNDF (YIN) Pitch Detection ─────────────────────────────────────────────
def detect_pitch(f: np.ndarray, sample_rate: int,
                 bounds: tuple = (60, 1200), thresh: float = 0.10) -> float | None:
    """
    YIN algoritmasının CMNDF adımı ile pitch tespiti.
    f: float32 numpy array (tarayıcıdan gelen ham ses verisi)
    """
    W = len(f) // 2

    min_lag = max(2, int(sample_rate / bounds[1]))
    max_lag = min(len(f) - W - 1, int(sample_rate / bounds[0]))

    if min_lag >= max_lag:
        return None

    lags = np.arange(min_lag, max_lag)

    # ── 1. Adım: Difference Function (DF) ────────────────────────────────────
    window_data = f[:W]
    df_values = np.array([
        np.sum((window_data - f[lag: lag + W]) ** 2)
        for lag in lags
    ])

    # ── 2. Adım: CMNDF ───────────────────────────────────────────────────────
    cumsum_df    = np.cumsum(df_values)
    running_mean = cumsum_df / (np.arange(1, len(df_values) + 1))
    cmndf_vals   = df_values / (running_mean + 1e-20)

    # ── 3. Adım: Eşiğin altındaki ilk yerel minimum ──────────────────────────
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
            return None
        sample_lag = int(lags[min_idx])

    # ── 4. Adım: Parabolic interpolation ─────────────────────────────────────
    rel = sample_lag - min_lag
    if 0 < rel < len(cmndf_vals) - 1:
        y0, y1, y2 = cmndf_vals[rel - 1], cmndf_vals[rel], cmndf_vals[rel + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            delta      = (y0 - y2) / denom
            sample_lag = sample_lag + delta

    return float(sample_rate / sample_lag)


# ─── Sessizlik kontrolü ───────────────────────────────────────────────────────
def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data ** 2)))


# ─── WebSocket handler ────────────────────────────────────────────────────────
async def handle(websocket):
    """
    Tarayıcıdan binary mesaj olarak Float32Array gelir.
    Python bunu numpy array'e çevirir, pitch hesaplar, JSON döner.
    """
    print(f"[SERVER] Bağlantı: {websocket.remote_address}")
    try:
        async for message in websocket:
            # Tarayıcı Float32Array'i binary olarak gönderir
            if isinstance(message, bytes):
                # Her 4 byte = 1 float32 örnek
                n_samples = len(message) // 4
                data = np.array(struct.unpack(f'{n_samples}f', message), dtype=np.float32)
            else:
                # Beklenmedik metin mesajı — atla
                continue

            # Sessizlik kontrolü
            if rms(data) < 0.008:
                await websocket.send(json.dumps({"silent": True}))
                continue

            # Pitch tespiti
            pitch = detect_pitch(data, SAMPLE_RATE)

            if pitch and 60 < pitch < 1200:
                payload = {"silent": False, **freq_to_note(pitch)}
            else:
                payload = {"silent": True}

            await websocket.send(json.dumps(payload))

    except websockets.exceptions.ConnectionClosedOK:
        print("[SERVER] Bağlantı düzgün kapatıldı.")
    except Exception as e:
        print(f"[SERVER] Hata: {e}")


# ─── Sunucu ───────────────────────────────────────────────────────────────────
async def main():
    print(f"[SERVER] Başlatılıyor → ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(handle, WS_HOST, WS_PORT):
        print("[SERVER] Hazır — tarayıcı bağlantısı bekleniyor.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
