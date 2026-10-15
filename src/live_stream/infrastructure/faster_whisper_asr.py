"""Türkçe ASR: faster-whisper (CTranslate2), fine-tuning projesindeki model (varsayılan medium).

Toplu iş ve canlı altyazı AYNI sınıfı (ve aynı süreçte aynı model nesnesini) kullanır:
  transcribe()       toplu iş: VAD parçaları -> mutlak zamanlı kelimeler (konuşmacıya atanır)
  transcribe_text()  canlı altyazı: tek cümlenin sesi -> ekranda gösterilecek metin

Live-stream'den gelenler: GPU varsa CUDA, CUDA açılamazsa (Windows'ta eksik cuBLAS/cuDNN)
çökmeden CPU'ya düşme; ara metin için beam 1, kesin metin için önceki cümleler istem olarak.

fine-tuning projesindeki FasterWhisperAdapter'dan uyarlandı; düzeltilenler:
  1. Ayarlar transcribe() içinde genel Settings/ortam değişkeninden okunuyordu (gizli
     bağımlılık, testte değiştirilemez). Burada hepsi kurucuda verilir (WhisperParams).
  2. Toplu (batched) çözme hatası yakalanamıyordu: faster-whisper sonuçları tembel
     üretir (generator), hata transcribe() çağrısında değil döngüde çıkar; try bloğu
     boşa çalışıyordu. Burada sonuçlar try içinde listeye alınır, hata olursa tek tek
     çözmeye düşülür.
  3. Tekrar süzgeci noktalamaya bakıyordu ("evet," ile "evet" farklı sayılıyordu);
     burada yalnızca harf/rakamla karşılaştırılır.
  4. Kendi VAD'ı yerine hattın VAD parçaları kullanılır (clip_timestamps): konuşmacı
     ayrımı ve ASR aynı parçalara bakar, ses iki kez taranmaz.
  5. log_prob_threshold ve istem (initial_prompt): bkz. WhisperParams.

Model klasörü (çevrimdışı): models/faster-whisper-medium/ (ya da -small)/{model.bin, config.json,
tokenizer.json, vocabulary.txt}; scripts/download_models.py indirir.
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

from live_stream.application.config import AsrParams, ModelRef
from live_stream.application.ports import TextNormalizer
from live_stream.domain import VadSegment, Word

log = logging.getLogger(__name__)
SR = 16000


def _key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def drop_repeats(words: list[Word], max_repeat: int) -> list[Word]:
    """Aynı kelime art arda max_repeat'ten fazla gelirse fazlasını atar
    ("evet evet evet evet" -> "evet evet"): Whisper'ın döngüye girmesine karşı."""
    out: list[Word] = []
    run = 0
    for w in words:
        run = run + 1 if out and _key(out[-1].text) == _key(w.text) else 1
        if run <= max_repeat:
            out.append(w)
    return out


def _add_nvidia_dll_dirs() -> None:
    """Windows: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` ile gelen DLL'leri görünür
    yapar; yoksa "cublas64_12.dll / cudnn_ops64_9.dll bulunamadı" hatası alınır."""
    if sys.platform != "win32":
        return
    import site
    from pathlib import Path

    for root in [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()]]:
        for bin_dir in root.glob("nvidia/*/bin"):
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def device_candidates(device: str, compute_type: str) -> list[tuple[str, str]]:
    """Denenecek (cihaz, hesaplama türü) sırası. auto: önce GPU, olmazsa CPU."""
    out = []
    if device in ("auto", "cuda"):
        out.append(("cuda", compute_type if compute_type != "auto" else "int8_float16"))
    if device in ("auto", "cpu"):
        out.append(("cpu", compute_type if compute_type != "auto" else "int8"))
    if not out:
        raise ValueError(f"WHISPER_DEVICE '{device}' geçersiz: auto | cuda | cpu")
    return out


def load_whisper(folder: str, device: str, compute_type: str, cpu_threads: int, num_workers: int):
    """Modeli ilk uygun cihazda açar; (model, cihaz, hesaplama türü) döndürür.
    Isınma çıkarımı yapılır: CUDA DLL eksikliği gibi hatalar genelde ilk çıkarımda patlar."""
    from faster_whisper import WhisperModel

    last: Exception | None = None
    for dev, ct in device_candidates(device, compute_type):
        try:
            if dev == "cuda":
                _add_nvidia_dll_dirs()
            t0 = time.monotonic()
            model = WhisperModel(folder, device=dev, compute_type=ct, cpu_threads=cpu_threads,
                                 local_files_only=True, num_workers=num_workers)
            list(model.transcribe(np.zeros(SR, dtype=np.float32), beam_size=1, language="tr")[0])
            log.info("Whisper %s yüklendi: %s/%s (%.1f sn)", folder, dev, ct, time.monotonic() - t0)
            return model, dev, ct
        except Exception as e:  # GPU yok, sürücü/DLL eksik, bellek yetmedi...
            last = e
            log.warning("Whisper %s/%s üzerinde açılamadı: %s", dev, ct, e)
    raise RuntimeError(f"Whisper modeli yüklenemedi: {last}")


class FasterWhisperAsr:
    segmentation = "vad"  # Whisper sessizlik/müzikte metin uyduruyor: yalnızca konuşma parçaları
    supports_fast = True    # feedback profili: beam 1 (açgözlü), projedeki gibi
    supports_rescue = True  # güvenlik ağı: yazısız kalan konuşma aralıklarını yeniden çözer

    def __init__(self, model: ModelRef, params: AsrParams, normalizer: TextNormalizer,
                 language: str, whisper_model=None, batched=None):
        """whisper_model/batched: testler için sahte nesne verilebilir; verilmezse yüklenir."""
        self.name = model.name
        self.language = language
        self.params = params
        self.wp = params.whisper
        self.normalizer = normalizer
        self._drop = {_key(p) for p in params.drop_phrases}
        if self.wp.initial_prompt:  # istem sessiz yerde aynen yazılırsa süzülsün
            self._drop.add(_key(self.wp.initial_prompt))
        self.device = "test"
        if whisper_model is None:
            from pathlib import Path

            from faster_whisper import BatchedInferencePipeline

            if not (Path(model.folder) / "model.bin").exists():
                # klasör yoksa kütüphane adı Hugging Face'te arar: internetsiz sunucuda
                # anlaşılmaz bir hata yerine ne yapılacağı söylenir
                raise FileNotFoundError(
                    f"{model.folder}/model.bin yok: python scripts/download_models.py --tr {model.name}")
            whisper_model, self.device, _ = load_whisper(
                str(model.folder), self.wp.device, self.wp.compute_type, params.num_threads,
                self.wp.num_workers)
            batched = BatchedInferencePipeline(model=whisper_model)
        self.model = whisper_model
        self.batched = batched

    def rescue(self, segments: list[VadSegment]) -> list[Word]:
        """Güvenlik ağı: ilk geçişte hiç kelime çıkmayan konuşma aralıkları, tek tek ve
        sıcaklık geri çekilmesiyle (0,0 -> 0,6) yeniden çözülür. Projede geri çekilme her
        parçada açıktı (yavaş: bir parça 4 kez çözülebilir); burada yalnızca kaçan aralıklarda."""
        raw = self._one_by_one(segments, rescue=True)
        words = []
        for seg_text, seg_words in raw:
            if _key(seg_text) in self._drop:
                continue
            for w in seg_words:
                text = self.normalizer.normalize(w.word)
                if text:
                    p = getattr(w, "probability", None)
                    words.append(Word(text=text, start=float(w.start), end=float(max(w.start, w.end)),
                                      conf=round(float(p), 3) if p is not None else None))
        return drop_repeats(words, self.wp.max_repeat)

    def _options(self, fast: bool = False, rescue: bool = False) -> dict:
        wp = self.wp
        if rescue:
            return dict(language=self.language, task="transcribe", beam_size=wp.beam_size,
                        word_timestamps=True, condition_on_previous_text=False,
                        temperature=(0.0, 0.2, 0.4, 0.6), repetition_penalty=wp.repetition_penalty,
                        no_repeat_ngram_size=wp.no_repeat_ngram_size, no_speech_threshold=0.8,
                        compression_ratio_threshold=wp.compression_ratio_threshold,
                        log_prob_threshold=-1.0, initial_prompt=wp.initial_prompt, hotwords=wp.hotwords)
        return dict(language=self.language, task="transcribe", beam_size=1 if fast else wp.beam_size,
                    word_timestamps=True, condition_on_previous_text=False, temperature=0.0,
                    repetition_penalty=wp.repetition_penalty,
                    no_repeat_ngram_size=wp.no_repeat_ngram_size,
                    no_speech_threshold=wp.no_speech_threshold,
                    compression_ratio_threshold=wp.compression_ratio_threshold,
                    log_prob_threshold=wp.log_prob_threshold,
                    initial_prompt=wp.initial_prompt, hotwords=wp.hotwords)

    def transcribe(self, segments: list[VadSegment], fast: bool = False) -> list[Word]:
        if not segments:
            return []
        raw = None
        if self.batched is not None:
            try:
                raw = self._batched(segments, fast)
            except Exception as e:  # ör. bellek: tek tek çözmeye düş
                log.warning("toplu çözme başarısız (%s), parçalar tek tek çözülüyor", e)
        if raw is None:
            raw = self._one_by_one(segments, fast)
        words: list[Word] = []
        for seg_text, seg_words in raw:
            if _key(seg_text) in self._drop:
                continue
            for w in seg_words:
                text = self.normalizer.normalize(w.word)
                if text:
                    p = getattr(w, "probability", None)
                    words.append(Word(text=text, start=float(w.start), end=float(max(w.start, w.end)),
                                      conf=round(float(p), 3) if p is not None else None))
        return drop_repeats(words, self.wp.max_repeat)

    def transcribe_text(self, samples: np.ndarray, fast: bool = False, prompt: str | None = None,
                        normalizer: TextNormalizer | None = None) -> str:
        """Canlı altyazı: tek cümlenin sesi -> metin (kelime zamanı yok, daha hızlı).

        fast=True: ara metin (beam 1, sıcaklık geri çekilmesi yok).
        prompt: önceki cümleler; Whisper'a bağlam olur (özel adlar, konu tutarlılığı).
        normalizer: ekran biçimi (büyük harf ve noktalama korunur); verilmezse toplu iş biçimi.
        Uydurma süzgeci ve tekrar ayıklaması toplu işle aynıdır."""
        if samples.size == 0:
            return ""
        wp = self.wp
        result, _ = self.model.transcribe(
            samples, language=self.language, task="transcribe", vad_filter=False,
            beam_size=1 if fast else wp.beam_size,
            temperature=0.0 if fast else (0.0, 0.2, 0.4),
            initial_prompt=prompt or wp.initial_prompt, hotwords=wp.hotwords,
            condition_on_previous_text=False, without_timestamps=True,
            repetition_penalty=wp.repetition_penalty, no_repeat_ngram_size=wp.no_repeat_ngram_size,
            no_speech_threshold=wp.no_speech_threshold,
            compression_ratio_threshold=wp.compression_ratio_threshold,
            log_prob_threshold=wp.log_prob_threshold)
        norm = normalizer or self.normalizer
        parts = []
        for r in result:
            if _key(r.text) in self._drop or (prompt and _key(r.text) == _key(prompt)):
                continue  # uydurma kalıp ya da istemin aynen tekrarı
            text = norm.normalize(r.text)
            if text:
                parts.append(text)
        words = [Word(text=w, start=0.0, end=0.0) for w in " ".join(parts).split()]
        return " ".join(w.text for w in drop_repeats(words, wp.max_repeat))

    def _batched(self, segments: list[VadSegment], fast: bool = False):
        """Parçalar asıl zamanlarıyla tek bir diziye yerleştirilir; clip_timestamps ile
        yalnızca onlar çözülür ve kelime zamanları doğrudan mutlak gelir."""
        end = max(s.end for s in segments)
        audio = np.zeros(int(np.ceil(end * SR)) + 1, dtype=np.float32)
        clips = []
        for s in segments:
            i = int(round(s.start * SR))
            n = min(len(s.samples), len(audio) - i)
            audio[i:i + n] = s.samples[:n]
            clips.append({"start": i / SR, "end": (i + n) / SR})
        result, _ = self.batched.transcribe(audio, clip_timestamps=clips,
                                            batch_size=self.wp.batch_size, **self._options(fast))
        return [(r.text, list(r.words or [])) for r in result]  # list(): hatalar burada çıksın

    def _one_by_one(self, segments: list[VadSegment], fast: bool = False, rescue: bool = False):
        out = []
        for s in segments:
            result, _ = self.model.transcribe(s.samples, vad_filter=False, **self._options(fast, rescue))
            for r in result:
                out.append((r.text, [_Shifted(w, s.start) for w in (r.words or [])]))
        return out


class _Shifted:
    """Tek parça çözümünde zamanlar parçanın başına göredir: mutlak zamana çevirir."""

    __slots__ = ("word", "start", "end", "probability")

    def __init__(self, w, offset: float):
        self.word, self.start, self.end = w.word, w.start + offset, w.end + offset
        self.probability = getattr(w, "probability", None)
