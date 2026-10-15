"""İnce ayar betiklerinin GPU gerektirmeyen parçaları."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "finetune"))
from ortak import gate2, leak_check, model_text  # noqa: E402


def test_metin_model_bicimine():
    assert model_text("bác sĩ, [ừ] thuốc   loãng máu.") == "BÁC SĨ Ừ THUỐC LOÃNG MÁU"
    assert model_text("Tiểu") == model_text("Tiểu")  # NFC: iki kodlanış aynı


def test_sizinti():
    assert leak_check({"train": [("a", "k1")], "test": [("b", "k2")]}) == []
    assert leak_check({"train": [("a", "k1")], "test": [("a", "k9")]})        # aynı ses
    assert leak_check({"train": [("a", "k1")], "test": [("b", "k1")]})        # aynı kayıt


def test_kapi2():
    base = {"tibbi": {"WER": 10.0, "RTF": 0.03}, "genel": {"WER": 8.0}}
    ok, _ = gate2(base, {"tibbi": {"WER": 9.0, "RTF": 0.03}, "genel": {"WER": 8.3}})
    assert ok
    ok, why = gate2(base, {"tibbi": {"WER": 9.8, "RTF": 0.03}, "genel": {"WER": 8.0}})
    assert not ok and "göreli kazanç" in why[0]                               # kazanç az
    ok, _ = gate2(base, {"tibbi": {"WER": 8.0, "RTF": 0.03}, "genel": {"WER": 9.0}})
    assert not ok                                                             # genel konuşmayı unuttu
    ok, why = gate2({"tibbi": base["tibbi"]}, {"tibbi": {"WER": 8.0, "RTF": 0.03}})
    assert not ok and "genel" in why[-1]                                      # genel test zorunlu
