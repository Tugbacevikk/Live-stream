"""İnce ayar betiklerinin ortak parçaları (bağımlılıksız; testler de kullanır)."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

# Zipformer-68M'in sözlüğü BÜYÜK HARF, noktalamasız Vietnamca hecelerden oluşur
# (tokens.txt: "▁CHÚNG", "▁PHẢI"...). Eğitim metni aynı biçimde olmalı, yoksa model
# küçük harfi "bilinmeyen" görür ve ince ayar zarar verir.
_PUNCT = re.compile(r"[^\w\s]")


def model_text(text: str) -> str:
    """VietMed / kendi etiketlerin -> modelin metin biçimi (NFC, büyük harf, noktalamasız).
    [ ] içi (araya giren konuşmacı) sözcükleriyle kalır, işaretleri silinir."""
    text = unicodedata.normalize("NFC", text).upper()
    return " ".join(_PUNCT.sub(" ", text).split())


def audio_hash(path: str | Path) -> str:
    """Ses içeriğinin özeti: aynı kayıt iki bölüme (train/test) düşmesin."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def leak_check(splits: dict[str, list[tuple[str, str]]]) -> list[str]:
    """Bölümler arası sızıntı: aynı dosya ya da aynı kayıt (recording) iki bölümde mi?
    splits: bölüm -> [(dosya_ozeti, kayit_adi)]. Sorun listesi döndürür (boş = temiz)."""
    problems, seen_hash, seen_rec = [], {}, {}
    for split, items in splits.items():
        for h, rec in items:
            if h in seen_hash and seen_hash[h] != split:
                problems.append(f"aynı ses hem {seen_hash[h]} hem {split} bölümünde")
            seen_hash.setdefault(h, split)
            if rec and rec in seen_rec and seen_rec[rec] != split:
                problems.append(f"'{rec}' kaydı hem {seen_rec[rec]} hem {split} bölümünde")
            if rec:
                seen_rec.setdefault(rec, split)
    return problems


def gate2(base: dict, new: dict, min_gain_rel: float = 0.05, max_general_loss: float = 0.5,
          max_rtf_ratio: float = 1.2) -> tuple[bool, list[str]]:
    """Kapı 2: ince ayarlı model temel modelin yerine geçebilir mi?

    base/new: {"tibbi": {"WER":..,"RTF":..}, "genel": {"WER":..}} (WER yüzde).
    Kural: tıbbi testte WER en az %5 (göreli) düşmeli; genel konuşmada en fazla 0,5 puan
    kötüleşebilir (unutma); hız en fazla 1,2 kat yavaşlayabilir (aynı mimari: beklenmez).
    """
    reasons, ok = [], True
    b, n = base["tibbi"]["WER"], new["tibbi"]["WER"]
    gain = (b - n) / b if b else 0.0
    if gain < min_gain_rel:
        ok = False
        reasons.append(f"tıbbi WER %{b:.2f} -> %{n:.2f}: göreli kazanç %{gain * 100:.1f} < %{min_gain_rel * 100:.0f}")
    else:
        reasons.append(f"tıbbi WER %{b:.2f} -> %{n:.2f} (göreli %{gain * 100:.1f} iyileşme)")
    if "genel" in base and "genel" in new:
        d = new["genel"]["WER"] - base["genel"]["WER"]
        if d > max_general_loss:
            ok = False
            reasons.append(f"genel konuşmada WER {d:+.2f} puan kötüleşti (sınır {max_general_loss})")
        else:
            reasons.append(f"genel konuşma WER farkı {d:+.2f} puan")
    else:
        ok = False
        reasons.append("genel konuşma testi yok: unutma (catastrophic forgetting) ölçülmeden karar verilmez")
    rb, rn = base["tibbi"].get("RTF"), new["tibbi"].get("RTF")
    if rb and rn and rn > rb * max_rtf_ratio:
        ok = False
        reasons.append(f"hız: RTF {rb} -> {rn}")
    return ok, reasons
