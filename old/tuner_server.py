"""
Gitar Akort Uygulaması - Python Backend
Gereksinimler: pip install numpy websockets aiohttp

Tek port üzerinden hem HTTP (index.html, style.css) hem WebSocket (/ws) sunar.
Render ve benzeri bulut platformlarında sorunsuz çalışır.

Çalıştırma:
    python tuner_server.py

Ortam değişkenleri:
    PORT=8765   (Render otomatik tanımlar)
"""

import asyncio
import json
import os
import struct
import numpy as np
import websockets
from aiohttp import web

# ─── Ayarlar ──────────────────────────────────────────────────────────────────
PORT        = int(os.environ.get("PORT", 8765))
SAMPLE_RATE = 44100

# ─── Kromatik nota isimleri ───────────────────────────────────────────────────
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


# ─── Frekans → Nota + Cent ────────────────────────────────────────────────────
def freq_to_note(freq: float) -> dict:
    if freq <= 0:
        return {"note": "—", "octave": "", "cents": 0, "freq": 0.0}

    semitones_from_a4 = 12 * np.log2(freq / 440.0)
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


# ─── CMNDF (YIN) Pitch Detection ─────────────────────────────────────────────
def detect_pitch(f: np.ndarray, sample_rate: int,
                 bounds: tuple = (60, 1200), thresh: float = 0.10) -> float | None:
    W       = len(f) // 2
    min_lag = max(2, int(sample_rate / bounds[1]))
    max_lag = min(len(f) - W - 1, int(sample_rate / bounds[0]))

    if min_lag >= max_lag:
        return None

    lags        = np.arange(min_lag, max_lag)
    window_data = f[:W]

    # 1. Difference Function
    df_values = np.array([
        np.sum((window_data - f[lag: lag + W]) ** 2)
        for lag in lags
    ])

    # 2. CMNDF
    cumsum_df    = np.cumsum(df_values)
    running_mean = cumsum_df / (np.arange(1, len(df_values) + 1))
    cmndf_vals   = df_values / (running_mean + 1e-20)

    # 3. Eşik altındaki ilk yerel minimum
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

    # 4. Parabolic interpolation
    rel = sample_lag - min_lag
    if 0 < rel < len(cmndf_vals) - 1:
        y0, y1, y2 = cmndf_vals[rel - 1], cmndf_vals[rel], cmndf_vals[rel + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            sample_lag = sample_lag + (y0 - y2) / denom

    return float(sample_rate / sample_lag)


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data ** 2)))


# ─── WebSocket handler ────────────────────────────────────────────────────────
async def ws_handler(request):
    """aiohttp WebSocket endpoint — /ws"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    print(f"[WS] Bağlantı: {request.remote}")

    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            n       = len(msg.data) // 4
            data    = np.array(struct.unpack(f'{n}f', msg.data), dtype=np.float32)

            if rms(data) < 0.008:
                await ws.send_str(json.dumps({"silent": True}))
                continue

            pitch = detect_pitch(data, SAMPLE_RATE)

            if pitch and 60 < pitch < 1200:
                payload = {"silent": False, **freq_to_note(pitch)}
            else:
                payload = {"silent": True}

            await ws.send_str(json.dumps(payload))

        elif msg.type == web.WSMsgType.ERROR:
            print(f"[WS] Hata: {ws.exception()}")

    print("[WS] Bağlantı kapatıldı.")
    return ws


# ─── HTTP: statik dosyaları sun ───────────────────────────────────────────────
async def serve_index(request):
    return web.FileResponse("index.html")

async def serve_css(request):
    return web.FileResponse("style.css")


# ─── Uygulama ─────────────────────────────────────────────────────────────────
app = web.Application()
app.router.add_get("/",          serve_index)
app.router.add_get("/style.css", serve_css)
app.router.add_get("/ws",        ws_handler)


if __name__ == "__main__":
    print(f"[SERVER] http://0.0.0.0:{PORT}  üzerinde başlatılıyor...")
    web.run_app(app, host="0.0.0.0", port=PORT)
