"""Whisper ile dil tespiti: TÜM dillerin olasılık dağılımı.

sherpa-onnx'in hazır SpokenLanguageIdentification sınıfı yalnızca en olası dili
döndürür (olasılık vermez). 99 dil arasında serbest seçim, gerçek Vietnamca
seslerde Tay, Arapça gibi yanlış dillere kaydı. Bu yüzden aynı ONNX dosyaları
onnxruntime ile doğrudan çalıştırılıp dil token'larının olasılıkları alınır;
desteklenen dillere kısıtlama uygulama katmanında yapılır.

Akış: 30 sn log-mel (Whisper'ın kendi formülü) -> encoder -> decoder'a tek
<|startoftranscript|> token'ı -> ilk adımın logit'lerinden dil token'ları.
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

SR, N_FFT, HOP, N_MELS, N_SAMPLES = 16000, 400, 160, 80, 30 * 16000


def _hz_to_mel(f):
    """Slaney mel ölçeği (librosa varsayılanı, Whisper bununla eğitildi)."""
    f = np.asanyarray(f, dtype=np.float64)
    fsp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / fsp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / fsp)


def _mel_to_hz(m):
    m = np.asanyarray(m, dtype=np.float64)
    fsp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / fsp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), m * fsp)


def mel_filters(n_mels: int = N_MELS) -> np.ndarray:
    fft_f = np.linspace(0, SR / 2, 1 + N_FFT // 2)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(0), _hz_to_mel(SR / 2), n_mels + 2))
    fd, ramps = np.diff(pts), pts[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        w[i] = np.maximum(0, np.minimum(-ramps[i] / fd[i], ramps[i + 2] / fd[i + 1]))
    w *= (2.0 / (pts[2:n_mels + 2] - pts[:n_mels]))[:, None]
    return w.astype(np.float32)


_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def log_mel(audio: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Whisper log-mel: 30 sn'ye kesilir/doldurulur -> (1, n_mels, 3000)."""
    x = np.pad(audio[:N_SAMPLES], (0, max(0, N_SAMPLES - len(audio)))).astype(np.float32)
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    frames = x[np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]]
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=1)) ** 2
    m = np.log10(np.maximum(filters @ power[:-1].T, 1e-10))
    m = np.maximum(m, m.max() - 8.0)
    return ((m + 4.0) / 4.0)[None].astype(np.float32)


class WhisperLanguageDetector:
    def __init__(self, encoder: str, decoder: str, num_threads: int = 2):
        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = num_threads, 1
        p = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(encoder, o, providers=p)
        self.dec = ort.InferenceSession(decoder, o, providers=p)
        meta = self.enc.get_modelmeta().custom_metadata_map
        if meta.get("is_multilingual") != "1":
            raise ValueError("dil tespiti için çok dilli Whisper modeli gerekir (ör. tiny, .en değil)")
        self.codes = meta["all_language_codes"].split(",")
        self.tokens = np.array([int(t) for t in meta["all_language_tokens"].split(",")])
        self.sot = int(meta["sot"])
        self.cache_shape = (int(meta["n_text_layer"]), 1, int(meta["n_text_ctx"]),
                            int(meta["n_text_state"]))
        self.filters = mel_filters(int(meta.get("n_mels", N_MELS)))

    def probabilities(self, audio: np.ndarray) -> dict[str, float]:
        k, v = self.enc.run(None, {"mel": log_mel(audio, self.filters)})
        cache = np.zeros(self.cache_shape, dtype=np.float32)
        logits = self.dec.run(["logits"], {
            "tokens": np.array([[self.sot]], dtype=np.int64),
            "in_n_layer_self_k_cache": cache, "in_n_layer_self_v_cache": cache,
            "n_layer_cross_k": k, "n_layer_cross_v": v,
            "offset": np.array([0], dtype=np.int64)})[0][0, -1]
        lang_logits = logits[self.tokens].astype(np.float64)
        p = np.exp(lang_logits - lang_logits.max())
        p /= p.sum()
        return dict(zip(self.codes, p.tolist()))


class LazyDetector:
    """Kademenin pahalı basamağı: yalnızca gerçekten gerekince yüklenir."""

    def __init__(self, factory):
        self._factory, self._det = factory, None

    def load(self):
        if self._det is None:
            self._det = self._factory()
        return self._det

    def probabilities(self, audio: np.ndarray) -> dict[str, float]:
        return self.load().probabilities(audio)
