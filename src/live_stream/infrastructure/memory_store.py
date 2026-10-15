"""Bellekte iş deposu: testler ve tek dosyalık CLI kullanımı için (kalıcı değil)."""
from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timezone

from live_stream.domain import (
    NOTIFY_DEFAULT, AudioJob, Correction, JobRecord, JobStatus, LanguageDecision, Transcript,
)


def _now():
    return datetime.now(timezone.utc)


class MemoryJobStore:
    def __init__(self):
        self.jobs: dict[str, JobRecord] = {}
        self.results: dict[str, Transcript] = {}
        self.events: list[tuple[str, str]] = []  # (job_id, olay) — outbox yerine
        self._lock = threading.Lock()
        self._originals: dict = {}
        self._edits: dict = {}
        self._deleted: set = set()
        self.consents: dict[str, tuple[str, str, datetime]] = {}

    def create(self, job: AudioJob) -> str:
        with self._lock:
            if job.job_id in self.jobs:
                raise KeyError(f"iş zaten var: {job.job_id}")
            if job.external_id:
                for r in self.jobs.values():
                    if r.job.external_id == job.external_id:
                        return r.job.job_id
            self.jobs[job.job_id] = JobRecord(job, JobStatus.QUEUED, created_at=_now(), updated_at=_now())
            return job.job_id

    def _update(self, job_id: str, **kw) -> JobRecord | None:
        with self._lock:
            r = self.jobs.get(job_id)
            if r is None:
                # CLI'da iş önceden açılmadan da çalışılabilsin
                r = JobRecord(AudioJob(job_id, ""), JobStatus.QUEUED, created_at=_now())
            r = replace(r, updated_at=_now(), **kw)
            self.jobs[job_id] = r
            return r

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def _event(self, r: JobRecord, event: str) -> None:
        notify = r.job.notify if r.job.notify is not None else NOTIFY_DEFAULT
        if r.job.callback_url and event in notify:
            self.events.append((r.job.job_id, f"job.{event}"))

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        r = self._update(job_id, status=status, error=error)
        if status in (JobStatus.REJECTED, JobStatus.RUNNING):
            self._event(r, status.value)

    def fail(self, job_id: str, error: str) -> None:
        self._event(self._update(job_id, status=JobStatus.FAILED, error=error), "failed")

    def set_language(self, job_id: str, decision: LanguageDecision) -> None:
        self._update(job_id, decision=decision)

    def save(self, t: Transcript) -> None:
        t = replace(t, utterances=[replace(u, index=i) for i, u in enumerate(t.utterances)])
        self.results[t.job_id] = t
        self._event(self._update(t.job_id, status=JobStatus.DONE, decision=t.language, error=None), "done")

    def get_transcript(self, job_id: str) -> Transcript | None:
        t = self.results.get(job_id)
        if t is None:
            return None
        return replace(t, utterances=[u for u in t.utterances if (job_id, u.index) not in self._deleted])

    def reopen(self, job_id, num_speakers):
        r = self.jobs.get(job_id)
        if r is None or r.status not in (JobStatus.DONE, JobStatus.FAILED) or not r.job.audio_path:
            return None
        return self._update(job_id, status=JobStatus.QUEUED, error=None,
                            job=replace(r.job, num_speakers=num_speakers))

    def delete_utterance(self, job_id, idx):
        with self._lock:
            t, r = self.results.get(job_id), self.jobs.get(job_id)
            if (t is None or r is None or r.status != JobStatus.DONE or not 0 <= idx < len(t.utterances)
                    or (job_id, idx) in self._deleted):
                return False
            self._deleted.add((job_id, idx))
            self._edits.pop((job_id, idx), None)
            return True

    def update_utterance(self, job_id, idx, speaker=None, text=None):
        with self._lock:
            t, r = self.results.get(job_id), self.jobs.get(job_id)
            if (t is None or r is None or r.status != JobStatus.DONE or not 0 <= idx < len(t.utterances)
                    or (job_id, idx) in self._deleted):
                return None
            old = t.utterances[idx]
            new = replace(old, speaker=speaker if speaker is not None else old.speaker,
                          text=text if text is not None else old.text, edited=True)
            utts = list(t.utterances)
            utts[idx] = new
            self.results[job_id] = replace(t, utterances=utts)
            orig = self._originals.setdefault((job_id, idx), (old.speaker, old.text))
            self._edits[(job_id, idx)] = (orig, _now())
            return new

    def set_consent(self, job_id, by, use):
        r = self.jobs.get(job_id)
        if r is None or r.status != JobStatus.DONE:
            return False
        self.consents[job_id] = (by, use, _now())
        return True

    def corrections(self, limit=10_000, consented_only=False):
        out = []
        for (jid, idx), ((osp, otx), at) in self._edits.items():
            if consented_only and jid not in self.consents:
                continue
            u, r = self.results[jid].utterances[idx], self.jobs[jid]
            out.append(Correction(jid, r.job.audio_path, r.decision.code if r.decision else None, idx,
                                  u.start, u.end, u.speaker, u.text, osp, otx, at))
        return sorted(out, key=lambda c: c.edited_at, reverse=True)[:limit]

    def recent_jobs(self, limit: int = 20) -> list[JobRecord]:
        return sorted(self.jobs.values(), key=lambda r: r.created_at, reverse=True)[:limit]

    def delete_job(self, job_id: str) -> str | None:
        with self._lock:
            r = self.jobs.pop(job_id, None)
            self.results.pop(job_id, None)
        return r.job.audio_path if r else None

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.jobs.values():
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out

    # --- bakım ve ölçüm ---
    _DONE = (JobStatus.DONE, JobStatus.FAILED, JobStatus.REJECTED)

    def finished_with_audio_before(self, before):
        return [(j, r.job.audio_path) for j, r in self.jobs.items()
                if r.status in self._DONE and r.updated_at < before and r.job.audio_path]

    def clear_audio(self, job_id):
        r = self.jobs[job_id]
        self.jobs[job_id] = replace(r, job=replace(r.job, audio_path=""))

    def delete_finished_before(self, before):
        old = [j for j, r in self.jobs.items() if r.status in self._DONE and r.updated_at < before]
        for j in old:
            self.jobs.pop(j)
            self.results.pop(j, None)
        return len(old)

    def stale_running(self, before):
        return [j for j, r in self.jobs.items() if r.status == JobStatus.RUNNING and r.updated_at < before]

    def stale_queued(self, before):
        return [r for r in self.jobs.values() if r.status == JobStatus.QUEUED and r.updated_at < before]

    def touch(self, job_id):
        self._update(job_id)

    def recent_timings(self, limit=500):
        done = [self.results[j] for j, r in self.jobs.items() if r.status == JobStatus.DONE and j in self.results]
        return [(t.language.code, t.duration, dict(t.timings)) for t in done if t.timings][:limit]

    def outbox_stats(self, max_attempts):
        return {"gonderildi": 0, "bekliyor": len(self.events), "olu": 0}

    def dead_deliveries(self, max_attempts, limit=100):
        return []

    def retry_delivery(self, outbox_id):
        return False

    def recent_done(self, limit=500):
        done = sorted((r for r in self.jobs.values() if r.status == JobStatus.DONE),
                      key=lambda r: r.updated_at, reverse=True)[:limit]
        return [(self.results[r.job.job_id].duration if r.job.job_id in self.results else 0.0,
                 (r.updated_at - r.created_at).total_seconds()) for r in done]
