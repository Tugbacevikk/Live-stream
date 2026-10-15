"""Ortam değişkenlerinden ayarlar.

Sıra: gerçek ortam değişkeni > proje kökündeki .env dosyası > buradaki varsayılan.
Windows'ta baslat.bat ilk çalıştırmada .env.example'dan .env oluşturur.
"""
from __future__ import annotations

import os

try:  # .env varsa yükle (python-dotenv yoksa sessizce geç)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass
from dataclasses import dataclass, field
from pathlib import Path


def _list(v: str) -> list[str]:
    return [x.strip().lower() for x in v.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # development: API anahtarı ve http/özel ağ callback'i serbest; production: zorunlu/kısıtlı
    app_env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development").lower())
    api_key: str = field(default_factory=lambda: os.environ.get("API_KEY", "").strip())
    webhook_secret: str = field(default_factory=lambda: os.environ.get("WEBHOOK_SECRET", "").strip())
    # boş değilse callback yalnızca bu adreslere gidebilir (iç ağ adresleri de yazılabilir)
    webhook_allowed_hosts: list[str] = field(default_factory=lambda: _list(
        os.environ.get("WEBHOOK_ALLOWED_HOSTS", "")))
    database_url: str = field(default_factory=lambda: os.environ.get(
        "DATABASE_URL", "postgresql://asr:asr@localhost:5432/asr"))
    # yerel mod (baslat.bat) veritabanı; boşsa DATA_DIR içinde jobs.db (SQLite).
    # PostgreSQL de verilebilir: LOCAL_DATABASE_URL=postgresql://...
    local_database_url: str = field(default_factory=lambda: os.environ.get("LOCAL_DATABASE_URL", ""))
    redis_url: str = field(default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    models_dir: str = field(default_factory=lambda: os.environ.get("MODELS_DIR", "models"))
    data_dir: str = field(default_factory=lambda: os.environ.get("DATA_DIR", "data/uploads"))
    # Türkçe ve Vietnamca. Yalnızca biri istenirse LANGUAGES=tr (dil tespiti modeli yüklenmez).
    languages: list[str] = field(default_factory=lambda: _list(os.environ.get("LANGUAGES", "tr,vi")))
    tr_model: str | None = field(default_factory=lambda: os.environ.get("TR_MODEL") or None)
    vi_model: str | None = field(default_factory=lambda: os.environ.get("VI_MODEL") or None)
    # faster-whisper nerede çalışsın: auto (GPU varsa GPU, yoksa/açılamazsa CPU) | cuda | cpu.
    # Sunucu (4000 mesaj, yalnızca CPU) için cpu; kendi bilgisayarında canlı altyazı için auto.
    whisper_device: str = field(default_factory=lambda: os.environ.get("WHISPER_DEVICE", "auto").lower())
    # auto: GPU'da int8_float16, CPU'da int8
    whisper_compute_type: str = field(default_factory=lambda: os.environ.get(
        "WHISPER_COMPUTE_TYPE", "auto").lower())
    threads: int = field(default_factory=lambda: int(os.environ.get("ONNX_THREADS", "2")))
    # 0: konuşmacı sayısı sesten bulunur (istekte num_speakers verilirse o kullanılır)
    speakers: int = field(default_factory=lambda: int(os.environ.get("NUM_SPEAKERS", "0")))
    diarizer: str = field(default_factory=lambda: os.environ.get("DIARIZER", "cluster"))
    # Whisper'a ipucu olarak verilen özel terimler, boşlukla ayrılmış (ör. "YOLOv8 TÜBİTAK")
    hotwords: str | None = field(default_factory=lambda: os.environ.get("HOTWORDS") or None)
    # Vietnamca terim dosyası (Zipformer): satır başına bir terim, BÜYÜK HARF (hotwords/tibbi_vi.txt)
    vi_hotwords: str | None = field(default_factory=lambda: os.environ.get("VI_HOTWORDS") or None)
    task_time_limit: int = field(default_factory=lambda: int(os.environ.get("TASK_TIME_LIMIT", "1800")))
    max_audio_minutes: float = field(default_factory=lambda: float(os.environ.get("MAX_AUDIO_MINUTES", "60")))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_MB", "100")))
    # sunucudaki bir dosya yolunu doğrudan kabul et (toplu iş için; yalnızca güvenilir ağda aç)
    allow_local_paths: bool = field(default_factory=lambda: os.environ.get(
        "ALLOW_LOCAL_PATHS", "0") == "1")
    # worker açılırken önceden yüklenecek diller (ilk işte gecikme olmasın)
    warmup: list[str] = field(default_factory=lambda: _list(os.environ.get("WARMUP_LANGS", "")))
    # dil tespiti worker'ı: Whisper-small'ı da baştan yükle (yoksa ilk zor seste ~5-15 sn gecikme)
    warmup_lid: bool = field(default_factory=lambda: os.environ.get("WARMUP_LID", "0") == "1")

    # saklama: biten işlerin ses dosyası N saat, iş kaydı ve sonucu N gün sonra silinir (0 = silme)
    audio_retention_hours: int = field(default_factory=lambda: int(os.environ.get(
        "AUDIO_RETENTION_HOURS", "24")))
    result_retention_days: int = field(default_factory=lambda: int(os.environ.get(
        "RESULT_RETENTION_DAYS", "30")))
    # RUNNING'de bundan uzun kalan iş askıda sayılır; MAX_JOB_ATTEMPTS denemeden sonra FAILED
    processing_stale_sec: int = field(default_factory=lambda: int(os.environ.get(
        "PROCESSING_STALE_SEC", "1800")))
    # Aşırı yük koruması. Kuyrukta/işlemde bekleyen iş bu sayıyı aşınca yeni istek 503 +
    # Retry-After alır (sistem çökmeden "sonra tekrar dene" der). 0 = sınırsız.
    # Varsayılan 10000: beklenen 4000 mesajlık ani yükün 2,5 katı.
    max_pending_jobs: int = field(default_factory=lambda: int(os.environ.get("MAX_PENDING_JOBS", "10000")))
    # İstemci (API anahtarı ya da IP) başına dakikada en fazla istek; 0 = kapalı (varsayılan).
    # Kapalı, çünkü tek entegre sistem 4000 mesajı bir anda gönderebilir; açılırsa 429 alır.
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.environ.get(
        "RATE_LIMIT_PER_MINUTE", "0")))
    # İşlem profili (istekte profile= ile iş başına değiştirilebilir): full | feedback
    pipeline_profile: str = field(default_factory=lambda: os.environ.get("PIPELINE_PROFILE", "full"))
    # feedback profilinde bundan kısa seste konuşmacı ayırma atlanır (sn)
    pipeline_min_diarize_sec: float = field(default_factory=lambda: float(os.environ.get(
        "PIPELINE_MIN_DIARIZE_SEC", "10")))
    # araya sıkışmış kısa konuşmacıyı komşusuna ver (sn; 0 = kapalı). Bkz. FusionConfig.
    fusion_sandwich_max: float = field(default_factory=lambda: float(os.environ.get(
        "FUSION_SANDWICH_MAX", "0")))
    # callback'e hangi olaylarda bildirim gitsin (istekte notify= ile iş başına)
    notify_default: list[str] = field(default_factory=lambda: _list(os.environ.get(
        "NOTIFY_EVENTS", "done,failed,rejected")))
    # webhook teslimi en fazla bu kadar denenir (10 sn'den 1 saate artan aralıklarla), sonra "ölü"
    webhook_max_attempts: int = field(default_factory=lambda: int(os.environ.get(
        "WEBHOOK_MAX_ATTEMPTS", "8")))
    # QUEUED'da bundan uzun kalan iş, kuyruğu boşsa yeniden kuyruğa konur (tam mod, sn)
    queued_stale_sec: int = field(default_factory=lambda: int(os.environ.get("QUEUED_STALE_SEC", "300")))
    # Kontrol edilmiş sonuçları veri setine (test/eğitim) aktarma; varsayılan kapalı. Açıkken bile
    # her aktarım kontrol eden kişinin adı ve açık onay ister; önce bekleyen kuyruğa düşer.
    enable_dataset_export: bool = field(default_factory=lambda: os.environ.get(
        "ENABLE_DATASET_EXPORT", "0").lower() in ("1", "true", "evet"))
    dataset_dir: str = field(default_factory=lambda: os.environ.get("DATASET_DIR", "data/veri_seti"))
    max_job_attempts: int = field(default_factory=lambda: int(os.environ.get("MAX_JOB_ATTEMPTS", "3")))
    # modeller yüklenirken MANIFEST.json'daki sha256 ile karşılaştır
    verify_models: bool = field(default_factory=lambda: os.environ.get("VERIFY_MODELS", "1") == "1")

    # --- Canlı altyazı (/canli, ws://.../ws/live) ---
    # Canlı sayfanın varsayılan dili: tr | vi | auto (auto: ilk cümleden bulunur ve kilitlenir)
    live_language: str = field(default_factory=lambda: os.environ.get("LIVE_LANGUAGE", "tr").lower())
    # Canlı modun Türkçe modeli; boşsa TR_MODEL ile aynı (model bir kez yüklenir).
    # 4 GB GPU'da medium yetişmezse LIVE_MODEL=faster-whisper-small. Vietnamca: hep VI_MODEL.
    live_model: str | None = field(default_factory=lambda: os.environ.get("LIVE_MODEL") or None)
    live_vad: str = field(default_factory=lambda: os.environ.get("LIVE_VAD", "silero").lower())
    live_vad_threshold: float = field(default_factory=lambda: float(os.environ.get(
        "LIVE_VAD_THRESHOLD", "0.5")))
    end_silence_ms: int = field(default_factory=lambda: int(os.environ.get("END_SILENCE_MS", "700")))
    max_utterance_sec: float = field(default_factory=lambda: float(os.environ.get(
        "MAX_UTTERANCE_SEC", "12")))
    partial_interval_sec: float = field(default_factory=lambda: float(os.environ.get(
        "PARTIAL_INTERVAL_SEC", "1.2")))
    save_recordings: bool = field(default_factory=lambda: os.environ.get(
        "SAVE_RECORDINGS", "true").lower() in ("1", "true", "evet"))
    recordings_dir: str = field(default_factory=lambda: os.environ.get("RECORDINGS_DIR", "recordings"))
    # Oturum bitince kayıt toplu işe gönderilsin: konuşmacı ayrımlı, düzeltilebilir transkript
    live_auto_analyze: bool = field(default_factory=lambda: os.environ.get(
        "LIVE_AUTO_ANALYZE", "true").lower() in ("1", "true", "evet"))

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def local_db(self) -> str:
        return self.local_database_url or f"sqlite:///{Path(self.data_dir) / 'jobs.db'}"

    @property
    def concurrency(self) -> int:
        """Yerel mod (tek süreç, iş parçacıkları) için aynı anda işlenen ses sayısı."""
        return self.concurrency_plan()[0]

    def model_for(self, lang: str) -> str:
        from live_stream.infrastructure.catalog import DEFAULT_MODELS

        return self.models.get(lang) or DEFAULT_MODELS[lang]

    def concurrency_plan(self, queue: str | None = None) -> tuple[int, str]:
        """(iş sayısı, gerekçe). CONCURRENCY sayıysa o; 'auto' (varsayılan) ise çekirdek ve
        belleğin küçüğü.

        queue=None: yerel mod — tek süreç, modeller bir kez yüklenir, işler paylaşır.
        queue='vi'|'tr': tam mod — her süreç kendi modelini yükler (stream_worker --surec).
        Bellek: toplam belleğin %75'i (işletim sistemi ve diğer programlara pay bırakılır).
        """
        from live_stream.infrastructure.catalog import AUX_MEMORY_GB, memory_estimate

        v = os.environ.get("CONCURRENCY", "auto")
        if v != "auto":
            return max(1, int(v)), "CONCURRENCY ayarı"
        by_cpu = max(1, available_cpus() // max(1, self.threads))
        total = total_memory_gb()
        if total is None:
            return by_cpu, f"çekirdek: {available_cpus()} / {self.threads} (bellek okunamadı)"
        usable = total * 0.75
        langs = [queue] if queue else list(self.languages)
        est = {lang: memory_estimate(self.model_for(lang)) for lang in langs}
        if queue:  # her süreç: model + yardımcı modeller + bir iş
            fixed, per_job = 0.0, est[queue][0] + est[queue][1] + AUX_MEMORY_GB
        else:      # tek süreç: tüm modeller bir kez, her eşzamanlı iş en ağır dilin çalışma belleği
            fixed = AUX_MEMORY_GB + sum(f for f, _ in est.values())
            per_job = max(p for _, p in est.values())
        by_mem = max(1, int((usable - fixed) // per_job))
        n = min(by_cpu, by_mem)
        why = (f"çekirdek {available_cpus()}/{self.threads}={by_cpu}, bellek {total:.0f} GB "
               f"(%75: {usable:.1f}, sabit {fixed:.1f} + iş başı {per_job:.1f}) = {by_mem}")
        return n, why

    @property
    def models(self) -> dict[str, str]:
        return {k: v for k, v in (("vi", self.vi_model), ("tr", self.tr_model)) if v}


def total_memory_gb() -> float | None:
    """Bu sürecin kullanabileceği toplam bellek (GB): Docker sınırı, yoksa makinenin belleği."""
    try:
        v = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if v != "max":
            return int(v) / 2**30
    except (OSError, ValueError):
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    if os.name == "nt":  # psutil yoksa Windows API'si
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullTotalPhys / 2**30
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return None


def available_cpus() -> int:
    """Bu sürecin gerçekten kullanabileceği çekirdek sayısı.

    os.cpu_count() Docker'da ana makinenin TÜM çekirdeklerini döndürür; konteynere
    sınır konduysa (cpus: 4) cgroup'tan okunur, yoksa sürecin izinli çekirdekleri.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS
        return os.cpu_count() or 1
