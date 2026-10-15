"""Prometheus metrikleri (metin biçimi): GET /metrics/prometheus.

Fine-tuning projesindeki metriklerin karşılığı, bir farkla: aşama süreleri süreç belleğinde
değil veritabanında (jobs.timings) tutulur. Tam modda işler ayrı süreçlerde (stream_worker)
çalışır; süreç içi sayaçlar API sürecinde görünmezdi, veritabanındaki her süreçten görünür.
prometheus_client gerekmez (biçim düz metin).
"""
from __future__ import annotations

from live_stream.application.maintenance import latency_stats

STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
RATIO_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 5)


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


class _Out:
    def __init__(self):
        self.lines: list[str] = []

    def metric(self, name: str, kind: str, help_: str):
        self.lines += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}"]

    def sample(self, name: str, value, **labels):
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        self.lines.append(f"{name}{{{lab}}} {value}" if lab else f"{name} {value}")

    def histogram(self, name: str, values: list[float], buckets, **labels):
        for b in buckets:
            self.sample(f"{name}_bucket", sum(1 for v in values if v <= b), le=b, **labels)
        self.sample(f"{name}_bucket", len(values), le="+Inf", **labels)
        self.sample(f"{name}_sum", round(sum(values), 4), **labels)
        self.sample(f"{name}_count", len(values), **labels)


def prometheus_text(store, settings, queue_stats=None, window: int = 1000) -> str:
    o = _Out()
    counts = store.counts()
    o.metric("asr_jobs", "gauge", "Durumlara göre iş sayısı (veritabanında)")
    for status in ("queued", "running", "done", "failed", "rejected"):
        o.sample("asr_jobs", counts.get(status, 0), status=status)

    if queue_stats is not None:
        try:
            qs = queue_stats()
        except Exception:
            qs = {}
        o.metric("asr_queue_depth", "gauge", "Kuyrukta bekleyen (henüz alınmamış) mesaj")
        for q, v in qs.items():
            if "bekleyen" in v:
                o.sample("asr_queue_depth", v["bekleyen"], queue=q)
        o.metric("asr_queue_in_progress", "gauge", "İşlenmekte olan mesaj")
        for q, v in qs.items():
            if "islenmekte" in v:
                o.sample("asr_queue_in_progress", v["islenmekte"], queue=q)
        o.metric("asr_queue_oldest_seconds", "gauge", "Bekleyen en eski mesajın yaşı (sn)")
        for q, v in qs.items():
            if "en_eski_sn" in v:
                o.sample("asr_queue_oldest_seconds", v["en_eski_sn"], queue=q)
        o.metric("asr_dead_letter_jobs", "gauge", "Deneme hakkı bitip bırakılan iş (DLQ)")
        o.sample("asr_dead_letter_jobs", (qs.get("dlq") or qs.get("birakilan") or {}).get("toplam", 0))

    timings = store.recent_timings(window)
    o.metric("asr_stage_seconds", "histogram", f"Aşama süreleri (son {window} iş)")
    stages = sorted({k for _, _, t in timings for k in t})
    for st in stages:
        o.histogram("asr_stage_seconds", [t[st] for _, _, t in timings if st in t], STAGE_BUCKETS, stage=st)

    o.metric("asr_realtime_ratio", "histogram", "İşlem süresi / ses süresi (hat içi; hedef <= 1)")
    for lang in sorted({lang for lang, _, _ in timings}):
        vals = [t["toplam"] / d for lg, d, t in timings if lg == lang and d > 0 and "toplam" in t]
        o.histogram("asr_realtime_ratio", vals, RATIO_BUCKETS, language=lang)

    lat = latency_stats(store.recent_done(window))
    o.metric("asr_end_to_end_ratio", "gauge", "İstekten bitişe süre / ses süresi (kuyruk dahil)")
    for q in ("p50", "p95"):
        o.sample("asr_end_to_end_ratio", lat.get(f"oran_{q}", 0), quantile=q.replace("p", "0."))
    o.metric("asr_end_to_end_over_target", "gauge", "Son işlerden hedefi (oran 1) aşanlar")
    o.sample("asr_end_to_end_over_target", lat.get("hedef_disi", 0))

    wh = store.outbox_stats(settings.webhook_max_attempts)
    o.metric("asr_webhook_deliveries", "gauge", "Webhook teslimleri: gonderildi, bekliyor, olu")
    for k, v in wh.items():
        o.sample("asr_webhook_deliveries", v, state=k)
    return "\n".join(o.lines) + "\n"
