"""sherpa-onnx ASR adaptörleri (Türkçe için yedek seçenekler): Omnilingual CTC (hızlı), Whisper-turbo.

Varsayılan Türkçe model faster-whisper (faster_whisper_asr.py); bunlar TR_MODEL ile seçilir.

Hepsi aynı arayüzü gerçekler: VAD parçaları -> mutlak zamanlı kelimeler.
Zaman bilgisi modele göre değişir:
  - Omnilingual: her parçanın başlangıç zamanı var.
  - Whisper (sherpa-onnx): kelime zamanı YOK. Parça süresi kelimelere harf
    sayısıyla orantılı paylaştırılır (yaklaşık; VAD parçaları kısa olduğu için
    konuşmacı ataması çoğunlukla doğru kalır, ama hassasiyet düşer).
"""
from __future__ import annotations

import math

import sherpa_onnx

from live_stream.application.config import AsrParams, LanguageProfile, ModelRef
from live_stream.application.ports import TextNormalizer
from live_stream.domain import VadSegment, Word

_SEP = ("▁", " ")  # BPE'de "▁" ya da boşluk: yeni kelime başlıyor


def tokens_to_words(tokens: list[str], timestamps: list[float], seg_start: float,
                    seg_end: float, normalizer: TextNormalizer,
                    max_duration: float = 1.0, log_probs: list[float] | None = None) -> list[Word]:
    """Model parçalarını kelimeye birleştirir ve mutlak zaman verir.

    İki biçimi de anlar:
      - BPE (Zipformer): "▁BÁC", "▁S", "Ĩ"  -> parça başında ▁ yeni kelime
      - harf (Omnilingual): "m","e"," ","d" -> tek başına " " ayırıcı
    Zaman damgası VAD parçasının başına göredir; seg_start eklenir.
    Bitiş = sonraki kelimenin başlangıcı (en fazla max_duration sonra).
    log_probs verilirse (Zipformer) kelime güveni = exp(parçaların log olasılık ortalaması).
    """
    raw = _group_tokens(tokens, timestamps, seg_start, seg_end, max_duration, log_probs)
    out = []
    for text, start, end, mlp in raw:
        text = normalizer.normalize(text)
        if text:
            out.append(Word(text=text, start=start, end=end,
                            conf=round(math.exp(mlp), 3) if mlp is not None else None))
    return out


def _group_tokens(tokens, timestamps, seg_start, seg_end, max_duration, log_probs):
    """-> [(ham metin, başlangıç, bitiş, log olasılık ortalaması | None)]"""
    lps = log_probs if log_probs and len(log_probs) == len(tokens) else None
    words: list[list] = []  # [metin, başlangıç, log olasılık toplamı, sayısı]
    new_word = True
    for n, (tok, t) in enumerate(zip(tokens, timestamps)):
        if tok.strip(" ▁") == "":  # yalnızca ayırıcı
            new_word = True
            continue
        if tok.startswith(_SEP) or new_word or not words:
            words.append([tok.lstrip(" ▁"), seg_start + t, lps[n] if lps else 0.0, 1 if lps else 0])
            new_word = False
        else:
            words[-1][0] += tok
            if lps:  # sırayla toplama (Python sum() 3.12'den beri telafili; Rust'la aynı olsun)
                words[-1][2] += lps[n]
                words[-1][3] += 1
    out = []
    for i, (text, start, total, k) in enumerate(words):
        nxt = words[i + 1][1] if i + 1 < len(words) else seg_end
        out.append((text, start, max(start, min(nxt, start + max_duration)), total / k if k else None))
    return out


def spread_words(text: str, seg_start: float, seg_end: float,
                 normalizer: TextNormalizer) -> list[Word]:
    """Zaman bilgisi olmayan metni parça süresine harf sayısıyla orantılı yayar."""
    parts = normalizer.normalize(text).split()
    if not parts:
        return []
    total = sum(len(p) + 1 for p in parts)
    dur, t, out = seg_end - seg_start, seg_start, []
    for p in parts:
        d = dur * (len(p) + 1) / total
        out.append(Word(text=p, start=t, end=t + d))
        t += d
    return out


def batches(segments: list[VadSegment], max_seconds: float) -> list[list[VadSegment]]:
    """Sırayı koruyarak, toplam süresi max_seconds'ı geçmeyen gruplar (tek parça hep sığar)."""
    out, cur, total = [], [], 0.0
    for s in segments:
        d = s.end - s.start
        if cur and total + d > max_seconds:
            out.append(cur)
            cur, total = [], 0.0
        cur.append(s)
        total += d
    if cur:
        out.append(cur)
    return out


class _SherpaAsr:
    segmentation = "full"

    def __init__(self, model: ModelRef, params: AsrParams, normalizer: TextNormalizer):
        self.name = model.name
        self.params = params
        self.normalizer = normalizer
        self.recognizer = self._build(model, params)

    def _build(self, model: ModelRef, params: AsrParams):  # pragma: no cover - alt sınıfta
        raise NotImplementedError

    def transcribe(self, segments: list[VadSegment]) -> list[Word]:
        out: list[Word] = []
        for batch in batches(segments, self.params.batch_seconds):
            streams = []
            for seg in batch:
                st = self.recognizer.create_stream()
                st.accept_waveform(seg.sample_rate, seg.samples)
                streams.append(st)
            self.recognizer.decode_streams(streams)  # grup içinde toplu çözme: tek tek çağırmaktan hızlı
            for seg, st in zip(batch, streams):
                out += self._words(st.result, seg)
            del streams  # grubun belleği bir sonrakinden önce bırakılır
        return out

    def transcribe_text(self, samples, fast: bool = False, prompt: str | None = None,
                        normalizer: TextNormalizer | None = None) -> str:
        """Canlı altyazı: tek cümle -> metin. Bu modellerde hızlı kip ve istem yok; metin
        toplu işin biçimindedir (küçük harf, noktalamasız)."""
        return " ".join(w.text for w in self.transcribe([VadSegment(0.0, samples)]))

    def _words(self, r, seg: VadSegment) -> list[Word]:
        return tokens_to_words(list(r.tokens), list(r.timestamps), seg.start, seg.end,
                               self.normalizer, self.params.max_word_duration,
                               list(getattr(r, "ys_log_probs", []) or []))


class SherpaWhisperAsr(_SherpaAsr):
    segmentation = "vad"

    def __init__(self, model, params, normalizer, language: str):
        self.language = language
        self._drop = {self._key(p) for p in params.drop_phrases}
        super().__init__(model, params, normalizer)

    @staticmethod
    def _key(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _build(self, model, params):
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=model.path("encoder"), decoder=model.path("decoder"),
            tokens=model.path("tokens"), language=self.language, task="transcribe",
            num_threads=params.num_threads, provider="cpu")

    def _words(self, r, seg):
        if self._key(r.text) in self._drop:
            return []
        return spread_words(r.text, seg.start, seg.end, self.normalizer)


class SherpaOmnilingualAsr(_SherpaAsr):
    def _build(self, model, params):
        return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=model.path("model"), tokens=model.path("tokens"),
            num_threads=params.num_threads, provider="cpu")


def build_asr(profile: LanguageProfile, normalizer: TextNormalizer):
    engine = _build_asr(profile, normalizer)
    engine.segmentation = profile.segmentation
    return engine


def _build_asr(profile: LanguageProfile, normalizer: TextNormalizer):
    kind = profile.asr_model.kind
    if kind == "whisper":
        return SherpaWhisperAsr(profile.asr_model, profile.asr, normalizer, profile.code)
    if kind == "omnilingual-ctc":
        return SherpaOmnilingualAsr(profile.asr_model, profile.asr, normalizer)
    if kind == "faster-whisper":  # isteğe bağlı bağımlılık: yalnızca seçilince yüklenir
        from live_stream.infrastructure.faster_whisper_asr import FasterWhisperAsr

        return FasterWhisperAsr(profile.asr_model, profile.asr, normalizer, profile.code)
    raise ValueError(f"bilinmeyen model türü: {kind}")
