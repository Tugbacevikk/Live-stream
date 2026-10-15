"""İş deposu (SQLAlchemy): aynı senaryolar SQLite'ta ve PostgreSQL'de.

    pytest tests/db                                                   # yalnız SQLite
    TEST_DATABASE_URL=postgresql://postgres@localhost/asr_test pytest tests/db   # ikisi
PostgreSQL testleri her biri kendi şemasında çalışır, sonra silinir.
"""
import os
import threading
import uuid
from datetime import timedelta

import httpx
import pytest

from live_stream.domain import (
    AudioJob, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance,
)
from live_stream.infrastructure.db import SqlAlchemyJobStore
from live_stream.infrastructure.webhook import OutboxDispatcher, verify_signature

PG = os.environ.get("TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgresql"])
def store(request, tmp_path):
    if request.param == "sqlite":
        s = SqlAlchemyJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")
        s.migrate()
        s.migrate()  # iki kez çalıştırmak zararsız
        yield s
        s.close()
        return
    if not PG:
        pytest.skip("TEST_DATABASE_URL yok")
    import psycopg

    schema = "t_" + uuid.uuid4().hex[:8]
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"CREATE SCHEMA {schema}")
    s = SqlAlchemyJobStore(f"{PG}{'&' if '?' in PG else '?'}options=-csearch_path%3D{schema}")
    s.migrate()
    s.migrate()
    yield s
    s.close()
    with psycopg.connect(PG, autocommit=True) as c:
        c.execute(f"DROP SCHEMA {schema} CASCADE")


def transcript(job_id):
    return Transcript(job_id=job_id, duration=12.5, model="zipformer-68m",
                      language=LanguageDecision("vi", 0.97, LanguageSource.DETECTED),
                      utterances=[Utterance("speaker_00", 0.1, 2.0, "xin chào"),
                                  Utterance("speaker_01", 2.5, 4.0, "chào bác sĩ")])


def test_yasam_dongusu(store):
    store.create(AudioJob("j1", "/a.wav", None, "http://alici/hook", 2))
    r = store.get("j1")
    assert r.status == JobStatus.QUEUED and r.job.num_speakers == 2 and r.decision is None
    assert r.created_at.tzinfo is not None
    store.set_language("j1", LanguageDecision("vi", 0.9, LanguageSource.DETECTED, True))
    assert store.get("j1").decision.uncertain is True
    store.set_status("j1", JobStatus.RUNNING)
    assert store.outbox_rows() == []  # ara durumlar bildirim üretmez
    assert [x.job.job_id for x in store.unfinished()] == ["j1"]
    store.save(transcript("j1"))
    store.save(transcript("j1"))  # tekrar işlenirse satırlar çoğalmaz
    assert store.get("j1").status == JobStatus.DONE and store.unfinished() == []
    t = store.get_transcript("j1")
    assert [u.text for u in t.utterances] == ["xin chào", "chào bác sĩ"]
    assert t.language.code == "vi" and t.model == "zipformer-68m" and t.duration == 12.5
    ob = store.outbox_rows()
    assert len(ob) == 2 and ob[0]["event"] == "job.done" and ob[0]["url"] == "http://alici/hook"
    assert ob[0]["payload"]["utterances"][1]["speaker"] == "speaker_01"
    assert store.recent_done()[0][0] == 12.5


def test_ayni_dis_kimlik_yeni_is_acmaz(store):
    assert store.create(AudioJob("a", "/a.wav", external_id="siparis-7")) == "a"
    assert store.create(AudioJob("b", "/b.wav", external_id="siparis-7")) == "a"
    assert store.create(AudioJob("c", "/c.wav")) == "c"
    assert store.get("b") is None and store.get("a").job.external_id == "siparis-7"


def test_ret_bildirimi_ve_bildirimsiz_is(store):
    store.create(AudioJob("r", "/a.wav", None, "http://alici/hook"))
    store.set_status("r", JobStatus.REJECTED, "desteklenmeyen dil")
    store.create(AudioJob("f", "/b.wav"))
    store.fail("f", "bozuk")
    assert [(o["job_id"], o["event"]) for o in store.outbox_rows()] == [("r", "job.rejected")]
    assert store.get_transcript("r") is None
    assert store.counts() == {"rejected": 1, "failed": 1}


def test_ara_hata_bildirim_uretmez_son_hata_uretir(store):
    store.create(AudioJob("x", "/a.wav", None, "http://alici/hook"))
    store.set_status("x", JobStatus.FAILED, "geçici hata")   # tekrar denenecek
    assert store.outbox_rows() == []
    store.fail("x", "en fazla deneme aşıldı")
    assert [(o["job_id"], o["event"]) for o in store.outbox_rows()] == [("x", "job.failed")]


def test_olmayan_is(store):
    store.set_status("yok", JobStatus.RUNNING)  # saklama süresi dolup silinmiş olabilir: sessiz
    with pytest.raises(KeyError):
        store.save(transcript("yok"))           # sonuç sessizce kaybolmasın


def test_webhook_gonderimi_basari_ve_tekrar(store):
    store.create(AudioJob("ok", "/a.wav", None, "http://alici/ok"))
    store.create(AudioJob("bozuk", "/b.wav", None, "http://alici/bozuk"))
    store.save(transcript("ok"))
    store.save(transcript("bozuk"))
    gelen = []

    def handler(req: httpx.Request):
        gelen.append((req.url.path, req.headers["X-Event"], req.headers["X-Job-Id"]))
        return httpx.Response(200 if req.url.path == "/ok" else 503)

    d = OutboxDispatcher(store, httpx.Client(transport=httpx.MockTransport(handler)), max_attempts=2)
    assert d.run_once() == 1 and ("/ok", "job.done", "ok") in gelen
    ob = {o["job_id"]: o for o in store.outbox_rows()}
    assert ob["ok"]["sent_at"] is not None
    assert ob["bozuk"]["sent_at"] is None and ob["bozuk"]["attempts"] == 1
    assert ob["bozuk"]["last_error"] == "HTTP 503"

    gelen.clear()
    assert d.run_once() == 0 and gelen == []  # bekleme süresi dolmadan tekrar denenmez
    assert store.claim_outbox(10, 8, timedelta(seconds=60)) == []
    store.mark_failed(ob["bozuk"]["id"], "HTTP 503", timedelta(0))  # bekleme bitti (2. deneme)
    assert d.run_once() == 0 and gelen == []  # max_attempts=2 doldu: bırakılır


def test_imzali_webhook(store):
    store.create(AudioJob("s", "/a.wav", None, "http://alici/ok"))
    store.save(transcript("s"))
    seen = []

    def handler(req: httpx.Request):
        seen.append(verify_signature("gizli", req.headers["X-Timestamp"], req.content,
                                     req.headers["X-Signature"]))
        return httpx.Response(200)

    OutboxDispatcher(store, httpx.Client(transport=httpx.MockTransport(handler)), secret="gizli").run_once()
    assert seen == [True]


def test_ayni_anda_iki_gonderici_ayni_satiri_almaz(store):
    for i in range(20):
        store.create(AudioJob(f"j{i}", "/a.wav", None, "http://alici/ok"))
        store.save(transcript(f"j{i}"))
    got, barrier = [], threading.Barrier(4)

    def claim():
        barrier.wait()
        got.extend(r["id"] for r in store.claim_outbox(10, 8, timedelta(seconds=60)))

    ts = [threading.Thread(target=claim) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(got) == len(set(got)) == 20


def test_esanli_yazma(store):
    """Yerel modda iş parçacıkları aynı anda yazar: kilitlenme ya da kayıp olmamalı."""
    errors = []

    def work(k):
        try:
            for i in range(10):
                jid = f"w{k}_{i}"
                store.create(AudioJob(jid, "/a.wav", None, "http://h/x"))
                store.set_status(jid, JobStatus.RUNNING)
                store.save(transcript(jid))
        except Exception as e:  # pragma: no cover - hata durumunda görünür olsun
            errors.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == [] and store.counts() == {"done": 40} and len(store.outbox_rows()) == 40


def test_sema_modellerle_ayni(store):
    """Göçler models.py ile birebir: yeni bir alan eklenip göçü unutulursa test kırılır."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from live_stream.infrastructure.db.models import Base

    with store.engine.connect() as c:
        assert compare_metadata(MigrationContext.configure(c, opts={"compare_type": True}),
                                Base.metadata) == []


LEGACY_SQLITE = """
CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, audio_path TEXT NOT NULL,
    requested_lang TEXT, callback_url TEXT, num_speakers INTEGER, lang_code TEXT,
    lang_confidence REAL, lang_source TEXT, lang_uncertain INTEGER, model TEXT, duration REAL,
    error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX jobs_status_idx ON jobs (status);
CREATE TABLE utterances (job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL, speaker TEXT NOT NULL, start_s REAL NOT NULL, end_s REAL NOT NULL,
    text TEXT NOT NULL, PRIMARY KEY (job_id, idx));
CREATE TABLE outbox (id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE, event TEXT NOT NULL,
    url TEXT NOT NULL, payload TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL, sent_at TEXT, last_error TEXT, created_at TEXT NOT NULL);
CREATE INDEX outbox_pending_idx ON outbox (next_attempt_at) WHERE sent_at IS NULL;
INSERT INTO jobs VALUES ('eski', 'done', '/a.wav', NULL, 'http://h/x', 2, 'vi', 1.0, 'istek', 0,
    'm', 3.0, NULL, '2026-10-12T09:00:00.000+00:00', '2026-10-12T09:00:02.500+00:00');
INSERT INTO utterances VALUES ('eski', 0, 'speaker_00', 0.0, 1.25, 'xin chào');
INSERT INTO outbox VALUES (7, 'eski', 'job.done', 'http://h/x', '{"status": "done"}', 0,
    '2026-10-12T09:00:02.500+00:00', NULL, NULL, '2026-10-12T09:00:02.500+00:00');
"""


def test_eski_sqlite_veritabani_aktarilir(tmp_path):
    """Alembic'ten önceki sürümün kurduğu jobs.db veri kaybı olmadan yeni şemaya geçer."""
    import sqlite3

    db = tmp_path / "jobs.db"
    with sqlite3.connect(db) as c:
        c.executescript(LEGACY_SQLITE)
    s = SqlAlchemyJobStore(f"sqlite:///{db}")
    s.migrate()
    s.migrate()
    r = s.get("eski")
    assert r.status == JobStatus.DONE and r.job.num_speakers == 2
    assert (r.updated_at - r.created_at).total_seconds() == 2.5 and r.created_at.tzinfo
    assert s.get_transcript("eski").utterances[0].text == "xin chào"
    ob = s.outbox_rows()
    assert ob[0]["id"] == 7 and ob[0]["payload"] == {"status": "done"}
    s.create(AudioJob("yeni", "/b.wav", None, "http://h/x"))
    s.fail("yeni", "x")
    assert [o["id"] for o in s.outbox_rows()] == [7, 8]  # sayaç kaldığı yerden
    s.close()


def test_duzeltme_ilk_hali_saklar_ve_sesi_korur(store):
    from datetime import timedelta as td

    from live_stream.infrastructure.db.models import utcnow

    store.create(AudioJob("d", "/up/d.wav", "vi"))
    assert store.update_utterance("d", 0, text="x") is None          # bitmemiş iş düzeltilmez
    store.save(transcript("d"))
    u = store.update_utterance("d", 1, text="chào bác sĩ ạ")
    assert u.text == "chào bác sĩ ạ" and u.edited and u.speaker == "speaker_01"
    store.update_utterance("d", 1, speaker="doktor")                 # ikinci düzeltme
    assert store.update_utterance("d", 9, text="x") is None          # olmayan satır
    t = store.get_transcript("d")
    assert [(x.speaker, x.edited) for x in t.utterances] == [("speaker_00", False), ("doktor", True)]
    (c,) = store.corrections()
    assert (c.original_text, c.original_speaker, c.text, c.speaker) == \
        ("chào bác sĩ", "speaker_01", "chào bác sĩ ạ", "doktor")      # ilk hal ilk düzeltmeden
    assert c.language == "vi" and c.audio_path == "/up/d.wav" and (c.start, c.end) == (2.5, 4.0)
    store.create(AudioJob("e", "/up/e.wav", "vi"))
    store.save(transcript("e"))
    later = utcnow() + td(days=2)
    assert [j for j, _ in store.finished_with_audio_before(later)] == ["e"]   # düzeltilenin sesi kalır


def test_bildirim_olaylari_profil_sureler_ve_silme(store):
    store.create(AudioJob("n", "/a.wav", "vi", "http://h/x", None, None, "feedback", ("running", "done")))
    store.set_status("n", JobStatus.RUNNING)
    t = transcript("n")
    from dataclasses import replace
    store.save(replace(t, profile="feedback", timings={"asr": 1.5, "toplam": 2.0}))
    assert [o["event"] for o in store.outbox_rows()] == ["job.running", "job.done"]
    r = store.get("n")
    assert r.job.profile == "feedback" and r.job.notify == ("running", "done")
    tt = store.get_transcript("n")
    assert tt.timings == {"asr": 1.5, "toplam": 2.0} and tt.profile == "feedback"
    store.create(AudioJob("q", "/b.wav", None, "http://h/x", notify=()))      # hiçbir bildirim
    store.fail("q", "x")
    assert [o["job_id"] for o in store.outbox_rows()] == ["n", "n"]
    assert store.delete_utterance("n", 0) and not store.delete_utterance("n", 0)
    assert [u.index for u in store.get_transcript("n").utterances] == [1]
    assert store.update_utterance("n", 0, text="x") is None                   # silinen düzeltilmez


def test_metrik_sorgulari_ve_olu_teslim(store):
    from dataclasses import replace

    store.create(AudioJob("m", "/a.wav", "vi", "http://h/x"))
    store.save(replace(transcript("m"), timings={"asr": 1.0, "toplam": 2.0}))
    assert store.recent_timings() == [("vi", 12.5, {"asr": 1.0, "toplam": 2.0})]
    (row,) = store.outbox_rows()
    for _ in range(3):
        store.mark_failed(row["id"], "HTTP 500", timedelta(0))
    assert store.outbox_stats(3) == {"gonderildi": 0, "bekliyor": 0, "olu": 1}
    assert [d["id"] for d in store.dead_deliveries(3)] == [row["id"]]
    assert store.retry_delivery(row["id"]) and store.outbox_stats(3)["bekliyor"] == 1
    assert store.claim_outbox(10, 3, timedelta(seconds=30))[0]["id"] == row["id"]


def test_yeniden_analiz_icin_acma(store):
    store.create(AudioJob("ra", "/a.wav", "vi"))
    assert store.reopen("ra", 2) is None                       # bitmedi
    store.save(transcript("ra"))
    r = store.reopen("ra", 3)
    assert r.status == JobStatus.QUEUED and r.job.num_speakers == 3 and r.decision.code == "vi"
    store.save(transcript("ra"))
    store.clear_audio("ra")
    assert store.reopen("ra", 2) is None                       # sesi silinmiş


def test_veri_seti_onayi(store):
    store.create(AudioJob("v", "/v.wav", "vi"))
    assert store.set_consent("v", "Dr. A", "test") is False      # bitmedi
    store.save(transcript("v"))
    store.update_utterance("v", 0, text="x")
    store.create(AudioJob("w", "/w.wav", "vi"))
    store.save(transcript("w"))
    store.update_utterance("w", 0, text="y")
    assert len(store.corrections()) == 2 and store.corrections(consented_only=True) == []
    assert store.set_consent("v", "Dr. A", "test") is True
    assert [c.job_id for c in store.corrections(consented_only=True)] == ["v"]
