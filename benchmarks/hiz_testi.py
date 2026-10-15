"""CPU hız ölçümü: Vietnamca ASR + VAD + diarization (sherpa-onnx).
Kullanım: python benchmarks/hiz_testi.py [model_dizini=models]
Üretim sunucusunda çalıştırıp RTF ve bellek değerlerini yol haritasına yazın.
"""
import json, os, resource, sys, time
import numpy as np
import soundfile as sf
import sherpa_onnx

D = sys.argv[1] if len(sys.argv) > 1 else "models"
Z68 = f"{D}/sherpa-onnx-zipformer-vi-int8-2025-04-20"
Z30 = f"{D}/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
WT = f"{D}/sherpa-onnx-whisper-turbo"
SR = 16000


def load(p):
    a, sr = sf.read(p, dtype="float32")
    assert sr == SR, sr
    return a


wavs = [load(f"{Z68}/test_wavs/{i}.wav") for i in range(3)]
gap = np.zeros(int(0.5 * SR), dtype=np.float32)


def build(sec):
    out, n, i = [], 0, 0
    while n < sec * SR:
        w = wavs[i % 3]
        out += [w, gap]
        n += len(w) + len(gap)
        i += 1
    return np.concatenate(out)[: sec * SR]


audio = {60: build(60), 120: build(120)}


def zipformer(path, enc, dec, joi, threads, method):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{path}/{enc}", decoder=f"{path}/{dec}", joiner=f"{path}/{joi}",
        tokens=f"{path}/tokens.txt", num_threads=threads, decoding_method=method,
        provider="cpu")


def whisper(threads):
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=f"{WT}/turbo-encoder.int8.onnx", decoder=f"{WT}/turbo-decoder.int8.onnx",
        tokens=f"{WT}/turbo-tokens.txt", language="vi", task="transcribe",
        num_threads=threads)


def vad_segments(a, threads=1):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = f"{D}/silero_vad.onnx"
    cfg.silero_vad.min_silence_duration = 0.25
    cfg.silero_vad.max_speech_duration = 20
    cfg.sample_rate = SR
    cfg.num_threads = threads
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=180)
    segs, w = [], cfg.silero_vad.window_size
    for i in range(0, len(a), w):
        vad.accept_waveform(a[i:i + w])
        while not vad.empty():
            segs.append(np.array(vad.front.samples, dtype=np.float32))
            vad.pop()
    vad.flush()
    while not vad.empty():
        segs.append(np.array(vad.front.samples, dtype=np.float32))
        vad.pop()
    return segs


def run_asr(rec, segs):
    streams = []
    for s in segs:
        st = rec.create_stream()
        st.accept_waveform(SR, s)
        streams.append(st)
    rec.decode_streams(streams)
    return " ".join(st.result.text.strip() for st in streams)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


res = {"cpu": os.popen("lscpu | grep 'Model name'").read().split(":")[1].strip(),
       "cores": os.cpu_count(), "rows": []}

# VAD süresi
for sec in (60, 120):
    t = time.time(); segs = vad_segments(audio[sec]); dt = time.time() - t
    res["rows"].append({"step": "VAD (Silero)", "audio_s": sec, "threads": 1,
                        "rtf": round(dt / sec, 4), "segments": len(segs)})
segs = {sec: vad_segments(audio[sec]) for sec in (60, 120)}

configs = [
    ("Zipformer-68M int8", lambda th, m: zipformer(Z68, "encoder-epoch-12-avg-8.int8.onnx",
     "decoder-epoch-12-avg-8.onnx", "joiner-epoch-12-avg-8.int8.onnx", th, m)),
    ("Zipformer-30M int8", lambda th, m: zipformer(Z30, "encoder.int8.onnx",
     "decoder.onnx", "joiner.int8.onnx", th, m)),
]
texts = {}
for name, mk in configs:
    for th in (1, 2):
        for m in ("greedy_search", "modified_beam_search"):
            t = time.time(); rec = mk(th, m); load_s = time.time() - t
            run_asr(rec, segs[60][:2])  # ısınma
            for sec in (60, 120):
                t = time.time(); txt = run_asr(rec, segs[sec]); dt = time.time() - t
                res["rows"].append({"step": f"ASR {name} {m}", "audio_s": sec, "threads": th,
                                    "rtf": round(dt / sec, 4), "load_s": round(load_s, 2)})
            texts[name] = [run_asr(rec, [w]) for w in wavs]
            del rec

# Whisper turbo (karşılaştırma), sadece 60 sn
for th in ((2,) if os.path.isdir(WT) else ()):
    t = time.time(); rec = whisper(th); load_s = time.time() - t
    run_asr(rec, [wavs[2]])
    t = time.time(); run_asr(rec, segs[60]); dt = time.time() - t
    res["rows"].append({"step": "ASR Whisper large-v3-turbo int8", "audio_s": 60, "threads": th,
                        "rtf": round(dt / 60, 4), "load_s": round(load_s, 2)})
    texts["Whisper turbo"] = [run_asr(rec, [w]) for w in wavs]
    del rec

# Diarization: pyannote segmentation-3.0 + wespeaker ResNet34 (pyannote 3.1 ile aynı model ailesi)
for emb_name, emb in (("wespeaker ResNet34", "wespeaker_en_voxceleb_resnet34_LM.onnx"),
                      ("3D-Speaker ERes2Net", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")):
    if not os.path.exists(f"{D}/{emb}"):
        continue
    for th in (1, 2):
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=f"{D}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
                num_threads=th),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=f"{D}/{emb}", num_threads=th),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=2),
            min_duration_on=0.3, min_duration_off=0.5)
        sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
        for sec in (60, 120):
            t = time.time(); r = sd.process(audio[sec]).sort_by_start_time(); dt = time.time() - t
            res["rows"].append({"step": f"Diarization pyannote-seg-3.0 + {emb_name}",
                                "audio_s": sec, "threads": th, "rtf": round(dt / sec, 4),
                                "segments": len(r)})
        del sd

res["peak_rss_mb"] = round(rss_mb())
res["texts"] = texts
print(json.dumps(res, ensure_ascii=False, indent=1))
