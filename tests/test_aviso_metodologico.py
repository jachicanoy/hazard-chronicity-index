"""Pruebas del aviso metodológico del dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exportacion import aviso_metodologico as am  # noqa: E402


def _irr() -> pd.DataFrame:
    estratos = ["Región Costa", "Región Sierra", "Región Oriente", "Amenaza flood",
                "Amenaza landslide", "Amenaza scour", "Amenaza debris_flow"]
    return pd.DataFrame({
        "estrato": estratos, "modelo": "M1_primario_ICEN_binario",
        "IRR": [3.4, 1.15, 1.09, 2.0, 1.2, 1.6, 0.46],
        "IC95_inf": [2.5, 0.9, 0.87, 1.5, 0.99, 1.3, 0.3], "IC95_sup": [4.7, 1.46, 1.36, 2.6, 1.47, 1.97, 0.7],
        "significativo_BH": [True, False, False, True, False, True, True],
    })


def test_inject_is_idempotent_and_preserves_bom(tmp_path):
    f = tmp_path / "a.html"
    f.write_bytes(b"\xef\xbb\xbf<html><body><header>H</header><main>x</main></body></html>")
    assert am.inject_banner(f, _irr()) is True
    primero = f.read_bytes()
    assert primero.startswith(b"\xef\xbb\xbf")                      # BOM conservado
    assert primero.count(am.MARK_START.encode()) == 1
    assert primero.index(b"</header>") < primero.index(am.MARK_START.encode())
    assert am.inject_banner(f, _irr()) is False                     # segunda corrida: sin cambios
    assert f.read_bytes().count(am.MARK_START.encode()) == 1


def test_banner_reflects_results_not_hardcoded():
    html_ = am.build_banner(_irr())
    assert "×3.40" in html_ and "menos eventos con El Niño" in html_ and "no detectable" in html_


def test_inject_falls_back_to_body_and_errors_without_anchor(tmp_path):
    g = tmp_path / "b.html"; g.write_text("<html><body><p>x</p></body></html>", encoding="utf-8")
    assert am.inject_banner(g, _irr()) is True
    h = tmp_path / "c.html"; h.write_text("<p>sin ancla</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        am.inject_banner(h, _irr())
    with pytest.raises(FileNotFoundError):
        am.inject_banner(tmp_path / "nada.html", _irr())
