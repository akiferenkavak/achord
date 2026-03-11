"""
Gitar Akort Uygulaması - Python Backend
Gereksinimler: pip install numpy pyaudio websockets

Yerel çalıştırma:
    python tuner_server.py

Sunucuda SSL ile çalıştırma:
    SSL_CERT=/etc/letsencrypt/live/alan.com/fullchain.pem \
    SSL_KEY=/etc/letsencrypt/live/alan.com/privkey.pem \
    WS_HOST=0.0.0.0 \
    python tuner_server.py

SSL sertifikası yoksa ücretsiz üretmek için:
    certbot certonly --standalone -d alan.com
"""

import asyncio
import json
import os
import ssl
import numpy as np
import websockets
import pyaudio

# ─── Ayarlar (ortam değişkenleriyle override edilebilir) ──────────────────────
SAMPLE_RATE = 44100
CHUNK       = 4096
CHANNELS    = 1
FORMAT      = pyaudio.paFloat32

WS_HOST = os.environ.get("WS_HOST", "localhost")
WS_PORT = int(os.environ.get("WS_PORT", 8765))

# SSL sertifika yolları (sunucuda tanımlıysa WSS aktif olur)
SSL_CERT = os.environ.get("SSL_CERT", "")   # örn: /etc/letsencrypt/live/x/fullchain.pem
SSL_KEY  = os.environ.get("SSL_KEY",  "")   # örn: /etc/letsencrypt/live/x/privkey.pem

# ─── Kromatik nota isimleri ───────────────────────────────────────────────────
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ─── Standart akort frekansları ───────────────────────────────────────────────
STANDARD_TUNING = {
    'E2': 82.41,
    'A2': 110.00,
    'D3': 146.83,
    'G3': 196.00,
    'B3': 246.94,
    'e4': 329.63,
}


# ─── Yardımcı: Frekans → Nota + Cent sapması ─────────────────────────────────
def freq_to_note(freq: float) -> dict:
    """
    Verilen frekansı en yakın nota + cent sapmasına çevirir.
    Cent: yarım ton = 100 cent. -50..+50 arası "doğru akort" bölgesi.
    """
    if freq <= 0:
        return {"note": "—", "octave": "", "cents": 0, "freq": 0.0}

    # A4 = 440 Hz referansına göre yarım ton sayısı
    semitones_from_a4 = 12 * np.log2(freq / 440.0)
    nearest_semitone  = round(semitones_from_a4)
    cents             = (semitones_from_a4 - nearest_semitone) * 100

    # Nota ismi ve oktav
    note_index = (nearest_semitone + 9) % 12   # A=9 offset
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
                 bounds: tuple = (50, 1200), thresh: float = 0.10) -> float | None:
    """
    YIN algoritmasının CMNDF adımı ile pitch tespiti.

    Parametreler
    ------------
    f           : zaman domainindeki ses verisi (float32 numpy array)
    sample_rate : örnekleme hızı (Hz)
    bounds      : (min_freq, max_freq) frekans aralığı
    thresh      : CMNDF eşiği; düşükse daha katı, yüksekse daha toleranslı

    Döndürür
    --------
    Tahmin edilen frekans (Hz) veya None (pitch tespit edilemezse)
    """
    W = len(f) // 2  # Pencere yarısı

    # Lag sınırları frekans aralığından hesaplanır
    min_lag = max(2, int(sample_rate / bounds[1]))
    max_lag = min(len(f) - W - 1, int(sample_rate / bounds[0]))

    if min_lag >= max_lag:
        return None

    lags = np.arange(min_lag, max_lag)

    # ── 1. Adım: Difference Function (DF) ────────────────────────────────────
    # DF(lag) = Σ [ f(t) - f(t + lag) ]²   için t = 0..W-1
    window_data = f[:W]
    df_values = np.array([
        np.sum((window_data - f[lag: lag + W]) ** 2)
        for lag in lags
    ])

    # ── 2. Adım: Cumulative Mean Normalized DF (CMNDF) ────────────────────────
    # CMNDF(lag) = DF(lag) / mean( DF(1..lag) )
    cumsum_df   = np.cumsum(df_values)
    running_mean = cumsum_df / (np.arange(1, len(df_values) + 1))
    cmndf_vals  = df_values / (running_mean + 1e-20)

    # ── 3. Adım: Eşiğin altındaki ilk yerel minimum ───────────────────────────
    sample_lag = None
    for i in range(1, len(cmndf_vals) - 1):
        if (cmndf_vals[i] < thresh
                and cmndf_vals[i] < cmndf_vals[i - 1]
                and cmndf_vals[i] < cmndf_vals[i + 1]):
            sample_lag = lags[i]
            break

    if sample_lag is None:
        # Eşik altına girilemedi → global minimumu al
        min_idx = int(np.argmin(cmndf_vals))
        if cmndf_vals[min_idx] > 0.35:   # çok gürültülüyse reddet
            return None
        sample_lag = int(lags[min_idx])

    # ── 4. Adım: Parabolic interpolation (hassas lag tahmini) ─────────────────
    rel = sample_lag - min_lag
    if 0 < rel < len(cmndf_vals) - 1:
        y0, y1, y2 = cmndf_vals[rel - 1], cmndf_vals[rel], cmndf_vals[rel + 1]
        denom = 2 * (2 * y1 - y0 - y2)
        if abs(denom) > 1e-10:
            delta = (y0 - y2) / denom
            sample_lag = sample_lag + delta

    return float(sample_rate / sample_lag)


# ─── Sessizlik kontrolü (RMS) ─────────────────────────────────────────────────
def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data ** 2)))


# ─── Ses okuma + WebSocket yayını ─────────────────────────────────────────────
async def audio_stream(websocket):
    """
    Mikrofonu açar, her CHUNK için pitch tespit eder,
    sonucu JSON olarak WebSocket üzerinden tarayıcıya gönderir.
    """
    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print(f"[SERVER] Bağlantı geldi: {websocket.remote_address}")
    print("[SERVER] Mikrofon açıldı, ses alınıyor...")

    try:
        while True:
            # Bloklayıcı PyAudio çağrısını asyncio'ya taşı
            raw = await asyncio.to_thread(
                stream.read, CHUNK, exception_on_overflow=False
            )
            data = np.frombuffer(raw, dtype=np.float32).copy()

            # Sessizlik filtresi
            if rms(data) < 0.008:
                await websocket.send(json.dumps({"silent": True}))
                continue

            pitch = detect_pitch(data, SAMPLE_RATE)

            if pitch and 60 < pitch < 1200:
                note_info = freq_to_note(pitch)
                payload   = {
                    "silent": False,
                    **note_info,
                }
            else:
                payload = {"silent": True}

            await websocket.send(json.dumps(payload))

    except websockets.exceptions.ConnectionClosedOK:
        print("[SERVER] Bağlantı kapatıldı.")
    except Exception as e:
        print(f"[SERVER] Hata: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("[SERVER] Mikrofon kapatıldı.")


# ─── WebSocket sunucusu ───────────────────────────────────────────────────────
async def main():
    ssl_ctx = None

    if SSL_CERT and SSL_KEY:
        # Sunucuda SSL sertifikası varsa WSS (güvenli) modda başlat
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=SSL_CERT, keyfile=SSL_KEY)
        proto = "wss"
    else:
        proto = "ws"

    print(f"[SERVER] WebSocket sunucusu başlatılıyor → {proto}://{WS_HOST}:{WS_PORT}")

    async with websockets.serve(audio_stream, WS_HOST, WS_PORT, ssl=ssl_ctx):
        if proto == "wss":
            print(f"[SERVER] SSL aktif — tarayıcıdan bağlanın: wss://ALAN_ADINIZ:{WS_PORT}")
        else:
            print("[SERVER] Yerel mod (SSL yok). Tarayıcıda index.html'i açın.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
