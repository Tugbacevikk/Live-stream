"""İş deposu + outbox, SQLAlchemy ile: SQLite (yerel mod) ve PostgreSQL (tam mod) tek kod.

Outbox neden? İş bitince müşteriye webhook gönderilmeli. Sonucu veritabanına yazıp
hemen ardından HTTP isteği atarsak, ikisinin arasında süreç çökerse bildirim kaybolur.
Bunun yerine sonuç ve "gönderilecek bildirim" satırı AYNI işlemde yazılır; ayrı bir
gönderici (dispatcher) outbox'taki satırları okuyup gönderir. Sonuç: bildirim en az
bir kez gider (alıcı X-Delivery-Id ile tekrarı ayıklar).

Şema models.py'de, değişiklikler Alembic göçüyle (migrate()). Eski sürümün elle
yazılmış SQL ile kurduğu veritabanı (alembic_version tablosu yok) migrate() içinde
tanınır: tablolar legacy_* adına çekilir, yeni şema kurulur, veri kopyalanır (SQLite'ta
zaman metinleri yeni biçime çevrilir), eski tablolar silinir. Tek işlemde: yarıda
kalırsa hiçbir şey değişmez.

SQLite notu: tek süreç + iş parçacıkları için (local_server). Yazma işlemleri
BEGIN IMMEDIATE ile başlar (iki iş parçacığı aynı anda yazmaya kalkınca kilitlenme
yerine sırayla bekler). Birden çok süreç (stream_worker) PostgreSQL ister.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, delete, event, func, inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from live_stream.application.serialization import transcript_to_dict
from live_stream.domain import (
    NOTIFY_DEFAULT, AudioJob, Correction, JobRecord, JobStatus, LanguageDecision, LanguageSource, Transcript, Utterance,
)
from live_stream.infrastructure.db.models import JobRow, OutboxRow, UtteranceRow, utcnow
from live_stream.infrastructure.db.url import normalize_url

MIGRATIONS = Path(__file__).parent / "migrations"
_TABLES = ("jobs", "utterances", "outbox")  # FK sırası: önce ebeveyn
_LEGACY_INDEXES = ("jobs_status_idx", "outbox_pending_idx")

# set_status ile bildirim üreten durumlar: ret kesindir (anında); RUNNING yalnızca istenirse.
# FAILED'in bildirimi yalnızca fail() ile (son deneme; ara denemeler bildirim üretmez).
_STATUS_EVENTS = {JobStatus.REJECTED: "rejected", JobStatus.RUNNING: "running"}
_FINISHED = (JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.REJECTED.value)


class SqlAlchemyJobStore:
    def __init__(self, url: str, pool_size: int = 5, echo: bool = False,
                 notify_default: tuple[str, ...] = NOTIFY_DEFAULT):
        self.notify_default = tuple(notify_default)
        self.url = normalize_url(url)
        self.dialect = self.url.split(":", 1)[0].split("+", 1)[0]
        if self.dialect == "sqlite":
            self.engine = create_engine(self.url, echo=echo,
                                        connect_args={"check_same_thread": False, "timeout": 30})
            _sqlite_setup(self.engine)
        else:
            self.engine = create_engine(self.url, echo=echo, pool_size=pool_size, max_overflow=2,
                                        pool_pre_ping=True)
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def close(self) -> None:
        self.engine.dispose()

    def ping(self) -> None:
        with self.engine.connect() as c:
            c.exec_driver_sql("SELECT 1")

    # --- şema ---
    def migrate(self) -> None:
        """alembic upgrade head. Eski (Alembic'siz) veritabanını önce tanır."""
        from alembic import command
        from alembic.config import Config

        cfg = Config()
        cfg.set_main_option("script_location", str(MIGRATIONS))
        with self.engine.begin() as conn:
            cfg.attributes["connection"] = conn
            tables = set(inspect(conn).get_table_names())
            legacy = "jobs" in tables and "alembic_version" not in tables
            if legacy:
                for t in _TABLES:
                    conn.exec_driver_sql(f"ALTER TABLE {t} RENAME TO legacy_{t}")
                for ix in _LEGACY_INDEXES:  # SQLite'ta indeks adları tabloyla taşınmaz, çakışır
                    conn.exec_driver_sql(f"DROP INDEX IF EXISTS {ix}")
            command.upgrade(cfg, "head")
            if legacy:
                self._copy_legacy(conn)

    def _copy_legacy(self, conn) -> None:
        """Eski tablolardaki veriyi yeni şemaya kopyalar, eski tabloları siler."""
        def ts(c):  # eski SQLite: "2026-10-13T10:00:00.123+00:00" -> "2026-10-13 10:00:00.123000"
            if self.dialect != "sqlite":
                return c
            return (f"CASE WHEN {c} LIKE '____-__-__T%' THEN "
                    f"replace(substr({c}, 1, 23), 'T', ' ') || '000' ELSE {c} END")

        conn.exec_driver_sql(
            "INSERT INTO jobs (id, status, audio_path, requested_lang, callback_url, num_speakers, "
            "lang_code, lang_confidence, lang_source, lang_uncertain, model, duration, error, "
            "created_at, updated_at) SELECT id, status, audio_path, requested_lang, callback_url, "
            "num_speakers, lang_code, lang_confidence, lang_source, lang_uncertain, model, duration, "
            f"error, {ts('created_at')}, {ts('updated_at')} FROM legacy_jobs")
        conn.exec_driver_sql(
            "INSERT INTO utterances (job_id, idx, speaker, start_s, end_s, text) "
            "SELECT job_id, idx, speaker, start_s, end_s, text FROM legacy_utterances")
        conn.exec_driver_sql(
            "INSERT INTO outbox (id, job_id, event, url, payload, attempts, next_attempt_at, sent_at, "
            f"last_error, created_at) SELECT id, job_id, event, url, payload, attempts, "
            f"{ts('next_attempt_at')}, {ts('sent_at')}, last_error, {ts('created_at')} FROM legacy_outbox")
        if self.dialect == "postgresql":  # kopyalanan id'lerden sonra sayaç devam etsin
            conn.exec_driver_sql("SELECT setval(pg_get_serial_sequence('outbox', 'id'), "
                                 "COALESCE((SELECT max(id) FROM outbox), 0) + 1, false)")
        for t in reversed(_TABLES):
            conn.exec_driver_sql(f"DROP TABLE legacy_{t}")

    @contextmanager
    def _tx(self):
        with self._session() as s, s.begin():
            yield s

    # --- yazma ---
    def create(self, job: AudioJob) -> str:
        """İşi kaydeder ve geçerli job_id'yi döndürür. external_id daha önce kullanıldıysa
        yeni iş açılmaz, var olanın kimliği döner (entegrasyonda tekrar gönderim güvenli)."""
        if job.external_id:
            found = self.find_by_external_id(job.external_id)
            if found:
                return found
        now = utcnow()
        try:
            with self._tx() as s:
                s.add(JobRow(id=job.job_id, status=JobStatus.QUEUED.value, audio_path=job.audio_path,
                             requested_lang=job.language, callback_url=job.callback_url,
                             num_speakers=job.num_speakers, external_id=job.external_id,
                             profile=job.profile,
                             notify=",".join(job.notify) if job.notify is not None else None,
                             created_at=now, updated_at=now))
        except IntegrityError:
            # iki istek aynı anda geldi: ikincisi benzersizlik kısıtına çarpar
            found = job.external_id and self.find_by_external_id(job.external_id)
            if not found:
                raise
            return found
        return job.job_id

    def find_by_external_id(self, external_id: str) -> str | None:
        with self._session() as s:
            return s.scalar(select(JobRow.id).where(JobRow.external_id == external_id))

    def _update_job(self, s: Session, job_id: str, must: bool = False, **values):
        """Satırı günceller; (callback_url, bildirim olayları) döndürür. İş yoksa (ör. saklama
        süresi dolup silindiyse) sessizce geçer; must=True ise KeyError (sonuç kaybolmasın)."""
        values["updated_at"] = utcnow()
        row = s.execute(update(JobRow).where(JobRow.id == job_id).values(**values)
                        .returning(JobRow.callback_url, JobRow.notify)).first()
        if row is None and must:
            raise KeyError(f"iş yok: {job_id}")
        if row is None or not row[0]:
            return None, ()
        events = tuple(row[1].split(",")) if row[1] is not None else self.notify_default
        return row[0], events

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._tx() as s:
            cb, events = self._update_job(s, job_id, status=status.value, error=error)
            event = _STATUS_EVENTS.get(status)
            if cb and event in events:
                self._outbox(s, job_id, f"job.{event}", cb,
                             {"job_id": job_id, "status": status.value, "error": error})

    def fail(self, job_id: str, error: str) -> None:
        with self._tx() as s:
            cb, events = self._update_job(s, job_id, status=JobStatus.FAILED.value, error=error)
            if cb and "failed" in events:
                self._outbox(s, job_id, "job.failed", cb,
                             {"job_id": job_id, "status": JobStatus.FAILED.value, "error": error})

    def set_language(self, job_id: str, d: LanguageDecision) -> None:
        with self._tx() as s:
            self._update_job(s, job_id, lang_code=d.code, lang_confidence=d.confidence,
                             lang_source=d.source.value, lang_uncertain=d.uncertain)

    def save(self, t: Transcript) -> None:
        d = t.language
        with self._tx() as s:
            cb, events = self._update_job(
                s, t.job_id, must=True, status=JobStatus.DONE.value, model=t.model,
                duration=t.duration, lang_code=d.code, lang_confidence=d.confidence,
                lang_source=d.source.value, lang_uncertain=d.uncertain, error=None,
                profile=t.profile, timings=t.timings or None, quality=t.quality or None)
            s.execute(delete(UtteranceRow).where(UtteranceRow.job_id == t.job_id))  # tekrar işlenirse
            s.add_all(UtteranceRow(job_id=t.job_id, idx=i, speaker=u.speaker, start_s=u.start,
                                   end_s=u.end, text=u.text, conf=u.conf,
                                   low_conf=",".join(map(str, u.low_conf)) or None)
                      for i, u in enumerate(t.utterances))
            if cb and "done" in events:
                self._outbox(s, t.job_id, "job.done", cb,
                             {"status": JobStatus.DONE.value, **transcript_to_dict(t)})

    def add_event(self, event: str, url: str, payload: dict, job_id: str | None = None) -> None:
        """İş durumundan bağımsız bildirim (ör. canlı oturum bitti). Aynı outbox'tan, aynı
        imza ve tekrar kurallarıyla gider; job_id verilirse iş silinince o da silinir."""
        with self._tx() as s:
            self._outbox(s, job_id, event, url, payload)

    @staticmethod
    def _outbox(s: Session, job_id: str | None, event: str, url: str, payload: dict) -> None:
        now = utcnow()
        s.add(OutboxRow(job_id=job_id, event=event, url=url, payload=payload,
                        next_attempt_at=now, created_at=now))

    # --- okuma ---
    def get(self, job_id: str) -> JobRecord | None:
        with self._session() as s:
            r = s.get(JobRow, job_id)
            return _record(r) if r else None

    def get_transcript(self, job_id: str) -> Transcript | None:
        with self._session() as s:
            j = s.get(JobRow, job_id)
            if j is None or j.status != JobStatus.DONE.value:
                return None
            rows = s.scalars(select(UtteranceRow).where(UtteranceRow.job_id == job_id,
                                                        UtteranceRow.deleted_at.is_(None))
                             .order_by(UtteranceRow.idx)).all()
            return Transcript(job_id=job_id, duration=j.duration or 0.0, model=j.model or "",
                              language=_decision(j), profile=j.profile or "full",
                              timings=dict(j.timings or {}), quality=dict(j.quality or {}),
                              utterances=[Utterance(r.speaker, r.start_s, r.end_s, r.text,
                                                    r.edited_at is not None, r.idx, r.conf,
                                                    _ints(r.low_conf) if r.edited_at is None else ())
                                          for r in rows])

    def set_consent(self, job_id: str, by: str, use: str) -> bool:
        """Veri setine aktarma onayı (kim, hangi amaçla). Bitmiş iş değilse False."""
        with self._tx() as s:
            j = s.get(JobRow, job_id)
            if j is None or j.status != JobStatus.DONE.value:
                return False
            j.consent_by, j.consent_at, j.consent_use = by, utcnow(), use
            return True

    def reopen(self, job_id: str, num_speakers: int | None) -> JobRecord | None:
        """Bitmiş işi yeniden analiz için kuyruğa hazırlar (QUEUED, yeni konuşmacı sayısı).
        Sesi silinmiş ya da bitmemiş işte None. Eski satırlar yeni sonuç kaydedilince değişir."""
        with self._tx() as s:
            j = s.get(JobRow, job_id)
            if j is None or j.status not in _FINISHED or j.status == JobStatus.REJECTED.value \
                    or not j.audio_path:
                return None
            j.status, j.num_speakers, j.error, j.updated_at = JobStatus.QUEUED.value, num_speakers, None, utcnow()
            s.flush()
            return _record(j)

    def delete_utterance(self, job_id: str, idx: int) -> bool:
        """Satırı sonuçtan kaldırır (uydurma satır). Kayıtta kalır: modelin hatası olarak
        ince ayar ve ölçüm için incelenebilir. Bitmiş iş değilse ya da satır yoksa False."""
        with self._tx() as s:
            j = s.get(JobRow, job_id)
            u = s.get(UtteranceRow, (job_id, idx))
            if j is None or u is None or j.status != JobStatus.DONE.value or u.deleted_at is not None:
                return False
            u.deleted_at = utcnow()
            return True

    def update_utterance(self, job_id: str, idx: int, speaker: str | None = None,
                         text: str | None = None) -> Utterance | None:
        with self._tx() as s:
            j = s.get(JobRow, job_id)
            u = s.get(UtteranceRow, (job_id, idx))
            if j is None or u is None or j.status != JobStatus.DONE.value or u.deleted_at is not None:
                return None
            if u.edited_at is None:  # modelin ilk hali bir kez saklanır
                u.original_text, u.original_speaker = u.text, u.speaker
            if speaker is not None:
                u.speaker = speaker
            if text is not None:
                u.text = text
            u.edited_at = utcnow()
            return Utterance(u.speaker, u.start_s, u.end_s, u.text, True, u.idx)

    def corrections(self, limit: int = 10_000, consented_only: bool = False) -> list[Correction]:
        """consented_only: yalnızca veri setine aktarma onayı verilmiş işlerden (ince ayar verisi)."""
        with self._session() as s:
            q = (select(UtteranceRow, JobRow.audio_path, JobRow.lang_code)
                 .join(JobRow, JobRow.id == UtteranceRow.job_id)
                 .where(UtteranceRow.edited_at.is_not(None), UtteranceRow.deleted_at.is_(None)))
            if consented_only:
                q = q.where(JobRow.consent_at.is_not(None))
            rows = s.execute(q.order_by(UtteranceRow.edited_at.desc()).limit(limit)).all()
            return [Correction(u.job_id, path, lang, u.idx, u.start_s, u.end_s, u.speaker, u.text,
                               u.original_speaker or u.speaker, u.original_text or u.text, u.edited_at)
                    for u, path, lang in rows]

    def recent_jobs(self, limit: int = 20) -> list[JobRecord]:
        with self._session() as s:
            rows = s.scalars(select(JobRow).order_by(JobRow.created_at.desc()).limit(limit)).all()
            return [_record(r) for r in rows]

    def delete_job(self, job_id: str) -> str | None:
        with self._tx() as s:
            path = s.scalar(delete(JobRow).where(JobRow.id == job_id).returning(JobRow.audio_path))
        return path

    def counts(self) -> dict[str, int]:
        with self._session() as s:
            return dict(s.execute(select(JobRow.status, func.count()).group_by(JobRow.status)).all())

    def unfinished(self) -> list[JobRecord]:
        """Süreç kapanırken yarım kalan işler (yeniden başlatınca tekrar kuyruğa alınır)."""
        with self._session() as s:
            rows = s.scalars(select(JobRow).where(JobRow.status.in_(
                (JobStatus.QUEUED.value, JobStatus.RUNNING.value))).order_by(JobRow.created_at)).all()
            return [_record(r) for r in rows]

    # --- bakım ve ölçüm ---
    def finished_with_audio_before(self, before: datetime) -> list[tuple[str, str]]:
        with self._session() as s:
            return [tuple(r) for r in s.execute(
                select(JobRow.id, JobRow.audio_path).where(
                    JobRow.status.in_(_FINISHED), JobRow.updated_at < before, JobRow.audio_path != "",
                    # düzeltilmiş işin sesi kayıt silinene kadar kalır (ince ayar verisi)
                    ~select(UtteranceRow.idx).where(UtteranceRow.job_id == JobRow.id,
                                                   UtteranceRow.edited_at.is_not(None)).exists()))]

    def clear_audio(self, job_id: str) -> None:
        with self._tx() as s:
            s.execute(update(JobRow).where(JobRow.id == job_id).values(audio_path=""))

    def delete_finished_before(self, before: datetime) -> int:
        with self._tx() as s:
            # işe bağlı olmayan eski bildirimler (canlı oturum sonu) de aynı sürede silinir
            s.execute(delete(OutboxRow).where(OutboxRow.job_id.is_(None), OutboxRow.created_at < before))
            # utterances/outbox: ON DELETE CASCADE (SQLite'ta foreign_keys=ON ile)
            return s.execute(delete(JobRow).where(JobRow.status.in_(_FINISHED),
                                                  JobRow.updated_at < before)).rowcount

    def stale_running(self, before: datetime) -> list[str]:
        with self._session() as s:
            return list(s.scalars(select(JobRow.id).where(JobRow.status == JobStatus.RUNNING.value,
                                                          JobRow.updated_at < before)))

    def stale_queued(self, before: datetime) -> list[JobRecord]:
        with self._session() as s:
            rows = s.scalars(select(JobRow).where(JobRow.status == JobStatus.QUEUED.value,
                                                  JobRow.updated_at < before)
                             .order_by(JobRow.created_at).limit(1000)).all()
            return [_record(r) for r in rows]

    def touch(self, job_id: str) -> None:
        with self._tx() as s:
            s.execute(update(JobRow).where(JobRow.id == job_id).values(updated_at=utcnow()))

    def recent_done(self, limit: int = 500) -> list[tuple[float, float]]:
        """(ses süresi, istekten bitişe geçen süre) — son biten işler."""
        with self._session() as s:
            rows = s.execute(select(JobRow.duration, JobRow.created_at, JobRow.updated_at)
                             .where(JobRow.status == JobStatus.DONE.value)
                             .order_by(JobRow.updated_at.desc()).limit(limit)).all()
        return [(float(d or 0), (u - c).total_seconds()) for d, c, u in rows]

    def recent_timings(self, limit: int = 500) -> list[tuple[str, float, dict]]:
        """(dil, ses süresi, aşama süreleri) — son biten işler (metrikler)."""
        with self._session() as s:
            rows = s.execute(select(JobRow.lang_code, JobRow.duration, JobRow.timings)
                             .where(JobRow.status == JobStatus.DONE.value, JobRow.timings.is_not(None))
                             .order_by(JobRow.updated_at.desc()).limit(limit)).all()
        return [(lang or "?", float(d or 0), dict(t or {})) for lang, d, t in rows]

    def outbox_stats(self, max_attempts: int) -> dict[str, int]:
        """Webhook teslimleri: gönderilen, bekleyen (tekrar denenecek), ölü (deneme hakkı bitti)."""
        with self._session() as s:
            sent = s.scalar(select(func.count()).select_from(OutboxRow).where(OutboxRow.sent_at.is_not(None)))
            dead = s.scalar(select(func.count()).select_from(OutboxRow).where(
                OutboxRow.sent_at.is_(None), OutboxRow.attempts >= max_attempts))
            pending = s.scalar(select(func.count()).select_from(OutboxRow).where(
                OutboxRow.sent_at.is_(None), OutboxRow.attempts < max_attempts))
        return {"gonderildi": sent or 0, "bekliyor": pending or 0, "olu": dead or 0}

    def dead_deliveries(self, max_attempts: int, limit: int = 100) -> list[dict]:
        """Deneme hakkı biten (ölü) teslimler, en yenisi önce."""
        with self._session() as s:
            rows = s.scalars(select(OutboxRow).where(OutboxRow.sent_at.is_(None),
                                                     OutboxRow.attempts >= max_attempts)
                             .order_by(OutboxRow.id.desc()).limit(limit)).all()
            return [_outbox_dict(r) for r in rows]

    def retry_delivery(self, outbox_id: int) -> bool:
        """Ölü (ya da bekleyen) teslimi sıfırlayıp hemen yeniden denenecek hale getirir."""
        with self._tx() as s:
            n = s.execute(update(OutboxRow).where(OutboxRow.id == outbox_id, OutboxRow.sent_at.is_(None))
                          .values(attempts=0, next_attempt_at=utcnow(), last_error=None)).rowcount
        return bool(n)

    # --- outbox gönderici için ---
    def claim_outbox(self, limit: int, max_attempts: int, lease: timedelta) -> list[dict]:
        """Gönderilecek satırları alır ve 'lease' süresince başka göndericilere görünmez yapar.
        Kilit HTTP isteği boyunca TUTULMAZ (yavaş alıcı veritabanını bekletmesin); gönderici
        çökerse satır lease bitince yeniden görünür. PostgreSQL'de SKIP LOCKED: iki gönderici
        aynı satırı almaz. SQLite'ta yazma zaten sırayla (BEGIN IMMEDIATE)."""
        now = utcnow()
        with self._tx() as s:
            q = (select(OutboxRow).where(OutboxRow.sent_at.is_(None), OutboxRow.next_attempt_at <= now,
                                         OutboxRow.attempts < max_attempts)
                 .order_by(OutboxRow.id).limit(limit))
            if self.dialect == "postgresql":
                q = q.with_for_update(skip_locked=True)
            rows = s.scalars(q).all()
            for r in rows:
                r.next_attempt_at = now + lease
            return [_outbox_dict(r) for r in rows]

    def mark_sent(self, outbox_id: int) -> None:
        with self._tx() as s:
            s.execute(update(OutboxRow).where(OutboxRow.id == outbox_id).values(
                sent_at=utcnow(), attempts=OutboxRow.attempts + 1, last_error=None))

    def mark_failed(self, outbox_id: int, error: str, retry_in: timedelta) -> None:
        with self._tx() as s:
            s.execute(update(OutboxRow).where(OutboxRow.id == outbox_id).values(
                attempts=OutboxRow.attempts + 1, last_error=error[:500],
                next_attempt_at=utcnow() + retry_in))

    def outbox_rows(self) -> list[dict]:
        with self._session() as s:
            return [_outbox_dict(r) for r in s.scalars(select(OutboxRow).order_by(OutboxRow.id))]


def _sqlite_setup(engine) -> None:
    @event.listens_for(engine, "connect")
    def _connect(dbapi_conn, _):
        dbapi_conn.isolation_level = None  # işlemleri pysqlite değil biz başlatırız
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    @event.listens_for(engine, "begin")
    def _begin(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE")


def _outbox_dict(r: OutboxRow) -> dict:
    return {"id": r.id, "job_id": r.job_id, "event": r.event, "url": r.url, "payload": r.payload,
            "attempts": r.attempts, "next_attempt_at": r.next_attempt_at, "sent_at": r.sent_at,
            "last_error": r.last_error, "created_at": r.created_at}


def _decision(r: JobRow) -> LanguageDecision | None:
    if not r.lang_code:
        return None
    return LanguageDecision(r.lang_code, float(r.lang_confidence or 0.0),
                            LanguageSource(r.lang_source), bool(r.lang_uncertain))


def _record(r: JobRow) -> JobRecord:
    job = AudioJob(r.id, r.audio_path, r.requested_lang, r.callback_url, r.num_speakers, r.external_id,
                   r.profile, tuple(r.notify.split(",")) if r.notify is not None else None)
    return JobRecord(job=job, status=JobStatus(r.status), decision=_decision(r), error=r.error,
                     created_at=r.created_at, updated_at=r.updated_at)


def _ints(v: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split(",")) if v else ()
