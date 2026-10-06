"""Pruebas unitarias del módulo asociacion_enso.

Se usan datos sintéticos con un efecto ENSO conocido para verificar que el
modelo lo recupera, y casos borde de la construcción del panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analisis import asociacion_enso as ae  # noqa: E402


def _enso(n_years: int = 11) -> pd.DataFrame:
    """Índice ENSO sintético: dos episodios contiguos de 6 meses."""
    rows = []
    for y in range(2015, 2015 + n_years):
        for m in range(1, 13):
            rows.append({"year": y, "month": m})
    df = pd.DataFrame(rows)
    df["es_icen"] = False
    df.loc[10:15, "es_icen"] = True
    df.loc[80:85, "es_icen"] = True
    df["icen_category"] = np.where(df["es_icen"], "El Niño - Débil", "Neutro")
    df["ICEN"] = np.where(df["es_icen"], 1.0, 0.0)
    return df


def _events_from_rate(enso: pd.DataFrame, irr: float, seed: int = 0) -> pd.DataFrame:
    """Eventos sintéticos NB con estacionalidad y efecto ENSO = ``irr``."""
    rng = np.random.default_rng(seed)
    base = 20 * (1 + 0.8 * np.sin(2 * np.pi * enso["month"] / 12))
    mu = base * np.where(enso["es_icen"], irr, 1.0)
    n = rng.negative_binomial(n=20, p=20 / (20 + mu))
    filas = []
    for (y, m), k in zip(zip(enso["year"], enso["month"]), n):
        filas += [{"year": y, "month": m, "province_code": 17, "hazard_type": "flood"}] * int(k)
    return pd.DataFrame(filas)


def test_panel_includes_zero_months():
    enso = _enso()
    ev = pd.DataFrame({"year": [2015], "month": [1]})
    panel = ae.build_monthly_panel(ev, enso)
    assert len(panel) == len(enso)          # ningún mes se descarta
    assert panel["n"].sum() == 1
    assert (panel["n"] == 0).sum() == len(enso) - 1


def test_detect_episodes_contiguous_blocks():
    enso = _enso()
    eps = ae.detect_episodes(enso, "es_icen")
    assert eps == [(10, 15), (80, 85)]


def test_model_recovers_known_irr():
    enso = _enso()
    ev = _events_from_rate(enso, irr=2.0)
    panel = ae.build_monthly_panel(ev, enso)
    x, y = ae.design_matrix(panel, "binary", "es_icen")
    fit = ae.fit_negbin(y, x)
    est = float(np.exp(fit.params["ElNino"]))
    assert 1.6 < est < 2.5
    assert fit.converged


def test_null_effect_not_significant():
    enso = _enso()
    ev = _events_from_rate(enso, irr=1.0, seed=3)
    panel = ae.build_monthly_panel(ev, enso)
    x, y = ae.design_matrix(panel, "binary", "es_icen")
    fit = ae.fit_negbin(y, x)
    assert fit.pvalues["ElNino"] > 0.01


def test_seasonality_control_reduces_crude_bias():
    """El IRR ajustado debe estar más cerca de 1 que el crudo cuando el
    episodio cae en temporada alta y no hay efecto real."""
    enso = _enso()
    enso["es_icen"] = False
    enso.loc[(enso["month"].isin([2, 3, 4])) & (enso["year"].isin([2016, 2020])), "es_icen"] = True
    enso["icen_category"] = np.where(enso["es_icen"], "El Niño - Débil", "Neutro")
    ev = _events_from_rate(enso, irr=1.0, seed=5)
    panel = ae.build_monthly_panel(ev, enso)
    crudo = ae.crude_rate_ratio(panel, "es_icen")
    x, y = ae.design_matrix(panel, "binary", "es_icen")
    ajustado = float(np.exp(ae.fit_negbin(y, x).params["ElNino"]))
    assert abs(np.log(ajustado)) < abs(np.log(crudo))


def test_lag_drops_initial_rows():
    enso = _enso()
    panel = ae.build_monthly_panel(_events_from_rate(enso, 1.5), enso)
    x0, _ = ae.design_matrix(panel, "binary", "es_icen", lag=0)
    x2, _ = ae.design_matrix(panel, "binary", "es_icen", lag=2)
    assert len(x2) == len(x0) - 2


def test_assign_region_and_missing_column():
    ev = pd.DataFrame({"province_code": [17, 20, 9]})
    out = ae.assign_region(ev, {"Sierra": [17], "Costa": [9]})
    assert out["region"].tolist()[0] == "Sierra" and pd.isna(out["region"].iloc[1])
    with pytest.raises(ae.AsociacionENSOError):
        ae.assign_region(pd.DataFrame({"x": [1]}), {"Costa": [9]})


def test_unknown_spec_raises():
    enso = _enso()
    panel = ae.build_monthly_panel(pd.DataFrame({"year": [2015], "month": [1]}), enso)
    with pytest.raises(ae.AsociacionENSOError):
        ae.enso_terms(panel, "inexistente", "es_icen")


def test_fdr_only_affects_primary_model():
    irr = pd.DataFrame({"modelo": ["M1", "M1", "S1"], "p_valor": [0.001, 0.04, 0.01]})
    out = ae.apply_fdr(irr, "M1")
    assert out.loc[out["modelo"] == "S1", "p_BH"].isna().all()
    assert (out.loc[out["modelo"] == "M1", "p_BH"] >= out.loc[out["modelo"] == "M1", "p_valor"]).all()


def test_classify_signal_rules():
    assert ae.classify_signal(3.4, True) == "Fuerte"
    assert ae.classify_signal(1.5, True) == "Moderada"
    assert ae.classify_signal(1.1, True) == "Débil"
    assert ae.classify_signal(0.46, True) == "Inversa"
    assert ae.classify_signal(3.4, False) == "No detectable"   # sin significancia no se etiqueta
    assert ae.classify_signal(float("nan"), True) == "No detectable"


def test_build_signal_table_regions_and_unevaluated():
    efci = pd.DataFrame({
        "canton_code": ["0901", "1701", "2001", "0901"],
        "canton_name_official": ["Guayaquil", "Quito", "San Cristobal", "Guayaquil"],
        "province_code": ["09", "17", "20", "09"],
        "hazard_type": ["TODOS", "TODOS", "TODOS", "debris_flow"],
    })
    irr = pd.DataFrame({
        "estrato": ["Región Costa", "Región Sierra"], "modelo": ["M1_primario_ICEN_binario"] * 2,
        "IRR": [3.4, 1.15], "IC95_inf": [2.5, 0.9], "IC95_sup": [4.7, 1.46],
        "significativo_BH": [True, False],
    })
    cfg = {"regiones": {"Costa": [9], "Sierra": [17]}, "senal_umbral_fuerte": 2.0, "senal_umbral_moderada": 1.2}
    out = ae.build_signal_table(efci, irr, cfg).set_index(["canton_code", "hazard_type"])
    assert out.loc[("0901", "TODOS"), "senal_enso"] == "Fuerte"
    assert out.loc[("1701", "TODOS"), "senal_enso"] == "No detectable"
    assert out.loc[("2001", "TODOS"), "senal_enso"] == "No evaluado"          # Galápagos
    assert out.loc[("0901", "debris_flow"), "senal_enso"] == "No evaluado"    # estrato omitido
