"""Model kataloğu: indirme adresleri, klasör adları, dosya rolleri, lisanslar.

Hem scripts/download_models.py hem de CLI buradan okur; model bilgisi tek yerde.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from live_stream.application.config import AsrParams, LanguageProfile, ModelRef

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Pack:
    url_path: str           # BASE'e göre
    target: str             # açılınca oluşan klasör ya da dosya adı
    license: str
    kind: str | None = None  # ASR ise adaptör türü
    files: dict[str, str] = field(default_factory=dict)
    keep: tuple[str, ...] = ()  # boş değilse açıldıktan sonra yalnızca bu dosyalar kalır
    hf_repo: str | None = None  # sherpa-onnx yerine Hugging Face deposundan (url_path kullanılmaz)
    hf_revision: str | None = None  # sabit sürüm (commit): aynı komut hep aynı dosyaları indirir


ASR_PACKS: dict[str, Pack] = {
    "whisper-turbo": Pack(
        "asr-models/sherpa-onnx-whisper-turbo.tar.bz2", "sherpa-onnx-whisper-turbo", "MIT",
        "whisper",
        {"encoder": "turbo-encoder.int8.onnx", "decoder": "turbo-decoder.int8.onnx",
         "tokens": "turbo-tokens.txt"}),
    # Fine-tuning projesindeki Türkçe model ailesi, CTranslate2 int8. Gerçek kelime zamanı verir
    # (sherpa-onnx Whisper vermiyor). Varsayılan medium (13 Ekim kararı); small daha hızlı seçenek.
    "faster-whisper-medium": Pack(
        "", "faster-whisper-medium", "MIT", "faster-whisper",
        {"model": "model.bin", "config": "config.json", "tokenizer": "tokenizer.json",
         "vocabulary": "vocabulary.txt"},
        hf_repo="Systran/faster-whisper-medium"),
    "faster-whisper-small": Pack(
        "", "faster-whisper-small", "MIT", "faster-whisper",
        {"model": "model.bin", "config": "config.json", "tokenizer": "tokenizer.json",
         "vocabulary": "vocabulary.txt"},
        hf_repo="Systran/faster-whisper-small"),
    "omnilingual-300m": Pack(
        "asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2",
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", "Apache-2.0",
        "omnilingual-ctc", {"model": "model.int8.onnx", "tokens": "tokens.txt"}),
}

AUX_PACKS: dict[str, Pack] = {
    "silero-vad": Pack("asr-models/silero_vad.onnx", "silero_vad.onnx", "MIT"),
    "pyannote-seg": Pack(
        "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "sherpa-onnx-pyannote-segmentation-3-0", "MIT"),
    "wespeaker": Pack(
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
        "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "WeSpeaker (Apache-2.0 kod; VoxCeleb ile eğitilmiş ağırlık lisansı doğrulanacak)"),
    # Konuşmacı imzası. 13 Ekim, VietMed'den kurulan 56 gerçek diyalog (sayı / cümle / RTF):
    #   CAM++ common      47/56  %91,7  0,030   <- seçilen
    #   ERes2Net base     40/56  %85,6  0,049
    #   ERes2NetV2        40/56  %87,1  0,103
    #   TitaNet large     38/56  %85,0  0,055
    #   ResNet293         CPU'da çok yavaş, ölçüm yarıda bırakıldı
    "campplus": Pack(
        # 'recongition' yazım hatası sherpa-onnx'in kendi adresinde var, düzeltmeyin
        "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "3D-Speaker / ModelScope (Apache-2.0 olduğu doğrulanacak)"),
    # gerçek modelli testlerin sesleri (tests/integration): Türkçe ve başka dillerde birer cümle
    "lid-test-wavs": Pack(
        "asr-models/spoken-language-identification-test-wavs.tar.bz2",
        "spoken-language-identification-test-wavs", "test sesleri (sherpa-onnx)"),
}

# Her dilin varsayılan modeli. Ölçüm (12 Ekim, 2 çekirdek):
#   Türkçe, 24 sentetik cümle:  whisper-turbo CER %3,0 | omnilingual-300m CER %3,8
#   Türkçe, 60 sn iki konuşmacılı diyalog (23 VAD parçası), tüm hat:
#     whisper-turbo 63,5 sn (ASR 53,7)  | omnilingual-300m 24,9 sn (ASR 12,1)
#   Whisper her parçayı 30 sn'ye doldurup işliyor; kısa konuşma sıralarında
#   maliyet parça sayısıyla artıyor ve kelime zamanı vermiyor. Omnilingual
#   gerçek zaman damgası veriyor, ama "varmı", "birkan" gibi bitişik yazım
#   hataları ve noktalamasız küçük harf çıktı var.
#   -> hız hedefini (1 dk ses <= 1 dk) karşılayan omnilingual-300m.
# Türkçe varsayılanı faster-whisper medium (13 Ekim, kullanıcı kararı). Bu ortamda indirilemediği
# için hızı ÖLÇÜLMEDİ. CPU'da small'dan ~2-3 kat yavaş olması beklenir; önceki hatta small,
# kullanıcının bilgisayarında oran 0,5-0,7 verdi. Ölçüm:
#   python benchmarks/kabul_testi.py --dil tr kayit.wav   (1 dk ses <= 1 dk mı?)
# Oran 1'i aşarsa sırasıyla: TR_MODEL=faster-whisper-small, sonra omnilingual-300m.
DEFAULT_MODELS = {"tr": "faster-whisper-medium"}
MODEL_CHOICES = {"tr": ["faster-whisper-medium", "faster-whisper-small", "omnilingual-300m",
                        "whisper-turbo"]}

# Bellek tahmini (GB): (model ağırlığı — bir süreçte bir kez, iş başına çalışma belleği).
# Model boyutundan (int8) ve kütüphane belgelerinden tahmin; faster-whisper'da iş başına değer,
# aynı anda çalışan her işçinin modelin kopyasını tutma olasılığına göre ihtiyatlı seçildi.
MODEL_MEMORY_GB: dict[str, tuple[float, float]] = {
    "omnilingual-300m": (0.4, 0.5),
    "whisper-turbo": (0.9, 0.8),
    "faster-whisper-small": (0.3, 0.5),
    "faster-whisper-medium": (0.8, 1.0),
}
# her süreçte bir kez: VAD + konuşmacı imzası + Python
AUX_MEMORY_GB = 0.6


def memory_estimate(model: str) -> tuple[float, float]:
    """Katalogda olmayan model: en büyük Türkçe modelinki kadar varsayılır (ihtiyatlı)."""
    return MODEL_MEMORY_GB.get(model, MODEL_MEMORY_GB["faster-whisper-medium"])


# Whisper'ın sessizlik/gürültüde uydurduğu, gözlenmiş cümleler. Parça YALNIZCA bunlardan
# ibaretse atılır (cümlenin içinde geçen "teşekkür ederim" etkilenmez). Kaynak: toplu iş
# projesindeki gözlem + Live-stream'in canlı kullanımda topladığı liste (Whisper'ın altyazı
# verisinden ezberlediği kalıplar; kayda ya da kelimeye özel düzeltme değil).
DROP_PHRASES = {"tr": (
    "İzlediğiniz için teşekkür ederim.", "İzlediğiniz için teşekkürler.", "Altyazı M.K.",
    "Altyazı", "Abone olmayı unutmayın.", "Abone olun.", "Videoyu beğenmeyi unutmayın.",
    "Bir sonraki videoda görüşmek üzere.", "Müzik", "Thanks for watching!",
    "Subtitles by the Amara.org community",
)}


def model_ref(key: str, models_dir: str | Path) -> ModelRef:
    p = ASR_PACKS[key]
    return ModelRef(name=key, kind=p.kind, folder=Path(models_dir) / p.target,
                    files=dict(p.files), license=p.license)


def language_profile(code: str, models_dir: str | Path, model: str | None = None,
                     **params) -> LanguageProfile:
    key = model or DEFAULT_MODELS[code]
    params.setdefault("drop_phrases", DROP_PHRASES.get(code, ()))
    if key not in MODEL_CHOICES.get(code, []):
        raise ValueError(f"'{key}' modeli '{code}' dili için tanımlı değil: {MODEL_CHOICES.get(code)}")
    ref = model_ref(key, models_dir)
    # Whisper sessizlik/müzikte metin uyduruyor: ona yalnızca VAD parçaları
    seg = "vad" if ref.kind in ("whisper", "faster-whisper") else "full"
    return LanguageProfile(code=code, asr_model=ref, asr=AsrParams(**params), segmentation=seg)
