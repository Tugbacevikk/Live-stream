from datetime import datetime, timedelta, timezone

from live_stream.application.maintenance import Maintenance, latency_stats
from live_stream.domain import (
    AudioJob, JobStatus, LanguageDecision, LanguageSource, Transcript,
)
from live_stream.infrastructure.db import SqlAlchemyJobStore
from live_stream.infrastructure.webhook import sign, verify_signature
from live_stream.interface.composition import safe_file_remover


def test_saklama_ve_askidaki_isler(tmp_path):
    up = tmp_path / "up"
    up.mkdir()
    store = SqlAlchemyJobStore(f"sqlite:///{tmp_path / 'j.db'}")
    store.migrate()
    disari = tmp_path / "kullanici.wav"
    disari.write_bytes(b"x")
    for jid, path in (("eski", up / "eski.wav"), ("yeni", up / "yeni.wav"), ("dis", disari)):
        if path.parent == up:
            path.write_bytes(b"x")
        store.create(AudioJob(jid, str(path), None, "http://a/h"))
        store.save(Transcript(jid, 1.0, [], "m", LanguageDecision("vi", 1.0, LanguageSource.REQUEST)))
    store.create(AudioJob("takili", str(up / "t.wav"), None, "http://a/h"))
    store.set_status("takili", JobStatus.RUNNING)

    now = datetime.now(timezone.utc)
    m = Maintenance(store, safe_file_remover(up), audio_hours=1, result_days=30, stale_sec=600,
                    now=lambda: now + timedelta(hours=2))
    out = m.run_once()
    assert not (up / "eski.wav").exists() and not (up / "yeni.wav").exists()
    assert disari.exists()                   # yükleme klasörü dışındaki dosyaya dokunulmaz
    assert out["ses_silindi"] == 2 and out["askida_kaldi"] == 1
    assert store.get("takili").status == JobStatus.FAILED
    assert store.get("eski").job.audio_path == "" and store.get_transcript("eski") is not None
    assert ("takili", "job.failed") in [(o["job_id"], o["event"]) for o in store.outbox_rows()]

    m2 = Maintenance(store, safe_file_remover(up), 1, 30, 600, now=lambda: now + timedelta(days=31))
    assert m2.run_once()["kayit_silindi"] == 4 and store.counts() == {}


def test_gecikme_istatistigi():
    s = latency_stats([(60, 20), (60, 30), (30, 45), (10, 5)])
    assert s["n"] == 4 and s["hedef_disi"] == 1 and s["oran_p95"] == 1.5


def test_webhook_imzasi():
    body = b'{"job_id":"x"}'
    sig = sign("s3cr3t", "1700000000", body)
    assert sig.startswith("sha256=")
    assert verify_signature("s3cr3t", "1700000000", body, sig, now=1700000100)
    assert not verify_signature("baska", "1700000000", body, sig, now=1700000100)
    assert not verify_signature("s3cr3t", "1700000000", body + b" ", sig, now=1700000100)
    assert not verify_signature("s3cr3t", "1700000000", body, sig, now=1700009999)  # eski


def test_kuyruga_girmemis_isler_bos_kuyrukta_yeniden_konur():
    from live_stream.infrastructure.memory_store import MemoryJobStore

    store = MemoryJobStore()
    store.create(AudioJob("kayip_vi", "/a.wav", "vi"))
    store.create(AudioJob("kayip_lid", "/b.wav"))
    store.create(AudioJob("sirada_tr", "/c.wav", "tr"))
    sent = []

    class Q:
        def enqueue_transcription(self, j, lang):
            sent.append((j, lang))

        def enqueue_detection(self, j):
            sent.append((j, "lid"))

    busy = {"tr"}                                                  # tr kuyruğu dolu: dokunma
    now = datetime.now(timezone.utc) + timedelta(minutes=10)
    m = Maintenance(store, lambda p: False, 0, 0, 600, now=lambda: now, queue=Q(),
                    queue_idle=lambda q: q not in busy, queued_stale_sec=300)
    assert m.run_once()["yeniden_kuyrukta"] == 2
    assert sorted(sent) == [("kayip_lid", "lid"), ("kayip_vi", "vi")]
    sent.clear()
    m.now = lambda: datetime.now(timezone.utc)                     # hemen sonraki tur: tekrar koymaz
    assert m.run_once()["yeniden_kuyrukta"] == 0 and sent == []


def test_tek_dilde_dili_belirsiz_is_o_dile_konur():
    """Tek dil varken tespit işçisi yok: dili yazılmamış iş 'lid' yerine Türkçe kuyruğuna."""
    from live_stream.infrastructure.memory_store import MemoryJobStore

    store = MemoryJobStore()
    store.create(AudioJob("dilsiz", "/b.wav"))
    sent = []

    class Q:
        def enqueue_transcription(self, j, lang):
            sent.append((j, lang))

        def enqueue_detection(self, j):  # pragma: no cover - çağrılmamalı
            sent.append((j, "lid"))

    now = datetime.now(timezone.utc) + timedelta(minutes=10)
    m = Maintenance(store, lambda p: False, 0, 0, 600, now=lambda: now, queue=Q(),
                    queue_idle=lambda q: True, queued_stale_sec=300, default_language="tr")
    assert m.run_once()["yeniden_kuyrukta"] == 1 and sent == [("dilsiz", "tr")]


def test_ise_bagli_olmayan_bildirim_imzali_gider_ve_alici_dogrular():
    """Gerçek HTTP alıcısı: 'live.ended' outbox'tan imzalı gider, alıcı imzayı doğrular."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from live_stream.infrastructure.db import SqlAlchemyJobStore
    from live_stream.infrastructure.webhook import OutboxDispatcher, verify_signature

    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((dict(self.headers), body))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    store = SqlAlchemyJobStore("sqlite://")
    store.migrate()
    payload = {"event": "live.ended", "session_id": "abc123", "lines": [{"id": 0, "text": "merhaba"}]}
    store.add_event("live.ended", f"http://127.0.0.1:{srv.server_port}/hook", payload)
    try:
        assert OutboxDispatcher(store, secret="gizli").run_once() == 1
    finally:
        srv.shutdown()
    headers, body = got[0]
    assert headers["X-Event"] == "live.ended" and headers["X-Session-Id"] == "abc123"
    assert headers["X-Job-Id"] == "" and headers["X-Delivery-Id"]
    assert verify_signature("gizli", headers["X-Timestamp"], body, headers["X-Signature"])
    assert json.loads(body) == payload
    assert store.outbox_rows()[0]["sent_at"] is not None


def test_ise_bagli_olmayan_eski_bildirim_saklama_suresinde_silinir():
    from live_stream.infrastructure.db import SqlAlchemyJobStore

    store = SqlAlchemyJobStore("sqlite://")
    store.migrate()
    store.add_event("live.ended", "http://h/x", {"session_id": "s"})
    assert store.delete_finished_before(datetime.now(timezone.utc) - timedelta(days=1)) == 0
    assert len(store.outbox_rows()) == 1                       # yeni: kalır
    store.delete_finished_before(datetime.now(timezone.utc) + timedelta(seconds=1))
    assert store.outbox_rows() == []                           # eski: silinir
