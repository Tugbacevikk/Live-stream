"""Dile özel metin düzeltme.

Türkçe (Whisper): çıktı büyük/küçük harf ve noktalama içerir. Sistem çıktısında
  (normalizer_for) küçük harfe çevrilir ve noktalama silinir (13 Ekim kararı:
  noktalama gerekmiyor, entegre sistem tek biçim alır). Canlı altyazı ekranı ise
  okunaklı olsun diye noktalamayı korur (live_normalizer).
  Küçük harf Türkçe kuralıyla (I -> ı, İ -> i); Python'un lower() bunu yanlış yapar.
  Kesme işareti kelime içindeyse kalır ("istanbul'da"), ondalık ayırıcı da ("3,5").
"""
from __future__ import annotations

import re
import unicodedata

# kelime içi kesme ('da) ve rakam arası nokta/virgül (3,5) korunur, diğer işaretler silinir
_PUNCT = re.compile(r"(?<=\w)['’](?=\w)|(?<=\d)[.,](?=\d)|([^\w\s])")


def _strip(text: str) -> str:
    return _PUNCT.sub(lambda m: " " if m.group(1) else m.group(0).replace("’", "'"), text)


# Kiril harfleri arasından Latin harflerine benzeyenler: model bir Türkçe kelimenin içine bunları
# karıştırabiliyor (14 Ekim: "nasыsыn" = nasılsın). Kelimede Latin harf varsa bunlar düzeltilir.
_HOMOGLYPH = str.maketrans({"ы": "ı", "і": "i", "о": "o", "а": "a", "е": "e", "с": "c", "р": "p",
                            "х": "x", "у": "y", "к": "k", "м": "m", "т": "t", "н": "n", "в": "v"})


def _is_latin(ch: str) -> bool:
    return unicodedata.name(ch, "").startswith("LATIN")


def _latin_word(w: str) -> bool:
    """Harflerin hepsi Latin alfabesinden mi (Türkçe harfler dahil; rakam serbest)."""
    return all(not ch.isalpha() or _is_latin(ch) for ch in w)


def _latin_fix(w: str) -> str | None:
    """Latin kelime -> aynen; karışık -> benzer harfler düzeltilir, yine karışıksa olduğu gibi
    bırakılır (gerçek kelime kaybolmasın); hiç Latin harf yoksa None (başka alfabeden uydurma)."""
    if _latin_word(w):
        return w
    if not any(ch.isalpha() and _is_latin(ch) for ch in w):
        return None
    return w.translate(_HOMOGLYPH)


def _clean(text: str, strip_punct: bool, latin_only: bool = False) -> str:
    if strip_punct:
        text = _strip(text)
    words = text.split()
    if latin_only:
        # Çok dilli modeller (Omnilingual 1600 dil, Whisper) müzik/gürültüde başka alfabeden
        # "kelime" uydurabiliyor (14 Ekim: Türkçe podcast sonunda Devanagari "कजतच").
        # Türkçe Latin alfabesiyle yazılır: başka alfabedeki kelime atılır.
        words = [f for f in map(_latin_fix, words) if f]
    return " ".join(words)


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


class TrNormalizer:
    def __init__(self, lowercase: bool = False, strip_punct: bool = False, latin_only: bool = False):
        self.lowercase, self.strip_punct, self.latin_only = lowercase, strip_punct, latin_only

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        if self.lowercase:
            text = turkish_lower(text)
        return _clean(text, self.strip_punct, self.latin_only)


def normalizer_for(lang: str):
    """Toplu iş çıktısının biçimi: küçük harf, noktalamasız (yeni dil eklenirse burada)."""
    return TrNormalizer(lowercase=True, strip_punct=True, latin_only=True)


def live_normalizer(lang: str):
    """Canlı altyazı: büyük harf ve noktalama korunur (ekranda okunur), yabancı alfabe süzülür."""
    return TrNormalizer(lowercase=False, strip_punct=False, latin_only=True)
