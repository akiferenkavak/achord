# Gitar Akort Uygulaması

Python backend + HTML frontend mimarisi.  
Pitch detection tamamen Python'da çalışır, sonuçlar WebSocket ile tarayıcıya aktarılır.

## Dosyalar

| Dosya | Görev |
|---|---|
| `tuner_server.py` | Python backend — mikrofon okur, CMNDF ile pitch tespit eder, WebSocket'ten yayınlar |
| `index.html` | Tarayıcı arayüzü — WebSocket'ten veri alır, gösterge ve metre'yi günceller |

## Kurulum

```bash
pip install numpy pyaudio websockets
```

> **Windows'ta PyAudio sorunu yaşarsanız:**
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

## Çalıştırma

1. Python sunucusunu başlatın:
   ```bash
   python tuner_server.py
   ```

2. `index.html` dosyasını tarayıcıda açın (çift tıkla veya `file://` ile).

3. **"Sunucuya Bağlan"** butonuna tıklayın.

4. Gitarınızı çalın — nota, frekans ve cent sapması anlık güncellenir.

## Nasıl Çalışır?

### Pitch Detection Algoritması (YIN/CMNDF)

```
Mikrofon → PyAudio buffer (4096 örnek)
    ↓
RMS sessizlik filtresi (< 0.008 → atla)
    ↓
1. Difference Function (DF)
   DF(lag) = Σ [f(t) - f(t+lag)]²
    ↓
2. Cumulative Mean Normalized DF (CMNDF)
   CMNDF(lag) = DF(lag) / running_mean(DF)
    ↓
3. Eşik altındaki (0.10) ilk yerel minimum → lag
    ↓
4. Parabolic interpolation → hassas lag
    ↓
pitch (Hz) = sample_rate / lag
    ↓
WebSocket JSON → Tarayıcı
```

### Cent Hesaplama

```
cents = 1200 × log₂(pitch / nearest_note_freq)
|cents| ≤ 5  → Tam akort (yeşil)
cents  > 5   → Telden gerilmeli (mavi)
cents  < -5  → Tel gevşek (turuncu)
```
