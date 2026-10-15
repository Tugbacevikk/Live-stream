import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from olcutler import cpwer, der, read_rttm, write_rttm  # noqa: E402

REF = [("A", 0.0, 10.0), ("B", 10.0, 20.0)]


def test_der_kusursuz_ve_etiket_adi_onemsiz():
    assert der(REF, [("x", 0.0, 10.0), ("y", 10.0, 20.0)], collar=0)["der"] == 0.0


def test_der_bilesenleri():
    # B'nin ilk 5 sn'si A diye etiketlenmiş: 5/20 karışan
    d = der(REF, [("x", 0.0, 15.0), ("y", 15.0, 20.0)], collar=0)
    assert d["karisan"] == pytest.approx(0.25, abs=1e-3) and d["kacirilan"] == 0
    # son 4 sn hiç bulunamamış: kaçırılan 4/20; fazladan 2 sn konuşma: yanlış alarm 2/20
    d = der(REF, [("x", 0.0, 10.0), ("y", 10.0, 16.0), ("x", 22.0, 24.0)], collar=0)
    assert d["kacirilan"] == pytest.approx(0.2, abs=1e-3)
    assert d["yanlis_alarm"] == pytest.approx(0.1, abs=1e-3)


def test_der_collar_sinir_hatasini_affeder():
    hyp = [("x", 0.0, 10.2), ("y", 10.2, 20.0)]
    assert der(REF, hyp, collar=0)["der"] > 0
    assert der(REF, hyp, collar=0.25)["der"] == 0.0


def test_cpwer():
    ref = {"A": "xin chào bác sĩ", "B": "chào anh"}
    assert cpwer(ref, {"s1": "chào anh", "s0": "xin chào bác sĩ"}) == 0.0
    # tüm sözler tek kişiye verilmiş: B'nin 2 kelimesi eksik + A'ya 2 fazla
    assert cpwer(ref, {"s0": "xin chào bác sĩ chào anh"}) == pytest.approx(4 / 6)


def test_rttm_gidis_donus(tmp_path):
    p = tmp_path / "a.rttm"
    write_rttm(p, "a", REF)
    assert read_rttm(p) == REF
