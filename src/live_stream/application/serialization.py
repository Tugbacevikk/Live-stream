"""Sonucun dışarıya giden JSON biçimi (API yanıtı ve webhook gövdesi aynı)."""
from __future__ import annotations

from live_stream.domain import JobRecord, LanguageDecision, Transcript


def decision_to_dict(d: LanguageDecision | None) -> dict | None:
    if d is None:
        return None
    return {"code": d.code, "confidence": d.confidence, "source": d.source.value,
            "uncertain": d.uncertain}


def transcript_to_dict(t: Transcript) -> dict:
    return {
        "job_id": t.job_id,
        "duration": round(t.duration, 3),
        "model": t.model,
        "language": decision_to_dict(t.language),
        "speakers": len({u.speaker for u in t.utterances} - {"BELIRSIZ"}),
        "text": t.text,
        "utterances": [{"index": u.index if u.index >= 0 else i, "speaker": u.speaker,
                        "start": round(u.start, 3), "end": round(u.end, 3), "text": u.text,
                        **({"edited": True} if u.edited else {}),
                        **({"confidence": u.conf} if u.conf is not None else {}),
                        **({"low_confidence_words": list(u.low_conf)} if u.low_conf else {})}
                       for i, u in enumerate(t.utterances)],
        **({"profile": t.profile} if t.profile != "full" else {}),
        **({"quality": t.quality} if t.quality else {}),
    }


def record_to_dict(r: JobRecord, transcript: Transcript | None = None) -> dict:
    out = {
        "job_id": r.job.job_id,
        "status": r.status.value,
        "requested_language": r.job.language,
        "num_speakers": r.job.num_speakers,
        "external_id": r.job.external_id,
        "profile": r.job.profile,
        "notify": list(r.job.notify) if r.job.notify is not None else None,
        "language": decision_to_dict(r.decision),
        "error": r.error,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }
    if transcript is not None:
        out["result"] = transcript_to_dict(transcript)
    return out
