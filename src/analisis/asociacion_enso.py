"""Asociación estadística entre El Niño y los eventos de amenaza.

Responde a una pregunta que las etapas previas solo abordaban de forma
descriptiva: ¿hay *más eventos* durante El Niño, una vez descontadas la
estacionalidad y la tendencia de reporte, y cómo varía esa asociación según
(a) la magnitud del episodio, (b) la región y (c) el tipo de amenaza?

Diseño estadístico (resumen; detalle y justificación en
``docs/methodology/15_enso_hazard_association.md``):

* Unidad de análisis: conteo mensual de eventos por estrato (132 meses).
* Modelo: regresión Binomial Negativa (NB2, enlace log) con efectos fijos de
  mes calendario (estacionalidad) y tendencia lineal (crecimiento del sistema
  de reporte). El Poisson no es apropiado: hay sobredispersión fuerte.
* Inferencia: errores estándar HAC (Newey-West, 3 rezagos) por la
  autocorrelación de los conteos mensuales; intervalos de bootstrap por
  bloques como verificación; corrección de Benjamini-Hochberg por pruebas
  múltiples.
* Robustez: leave-one-episode-out (6 episodios), índice ONI en vez de ICEN,
  estacionalidad armónica, rezagos de 0 a 3 meses, efectos fijos de año.

Limitación estructural que ningún modelo corrige: solo hay ~6 episodios El
Niño independientes en 2015-2025. Los intervalos reflejan incertidumbre de
muestreo mensual, pero la generalización a episodios futuros es
inherentemente limitada.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

ICEN_CATEGORIES_EL_NINO = ("El Niño - Débil", "El Niño - Moderado", "El Niño - Fuerte")


class AsociacionENSOError(RuntimeError):
    """Se lanza ante datos o configuración inválidos en la etapa de asociación ENSO–amenaza."""


@dataclass(frozen=True)
class FitResult:
    """Resultado resumido de un ajuste Binomial Negativo.

    Attributes:
        params: Coeficientes (escala log) indexados por nombre de término.
        conf_int: Intervalos de confianza 95 % (escala log).
        pvalues: Valores p bilaterales (errores HAC).
        alpha: Parámetro de dispersión NB2.
        aic: Criterio de Akaike del modelo.
        llf: Log-verosimilitud.
        converged: Si el optimizador reportó convergencia.
        resid_pearson: Residuos de Pearson (para diagnósticos).
        n_obs: Número de observaciones (meses) usadas.
    """

    params: pd.Series
    conf_int: pd.DataFrame
    pvalues: pd.Series
    alpha: float
    aic: float
    llf: float
    converged: bool
    resid_pearson: np.ndarray
    n_obs: int


# ---------------------------------------------------------------------------
# Construcción del panel mensual
# ---------------------------------------------------------------------------
def assign_region(events: pd.DataFrame, regiones: dict[str, list[int]]) -> pd.DataFrame:
    """Asigna región natural a cada evento según ``province_code``.

    Args:
        events: Tabla de eventos con columna ``province_code``.
        regiones: Mapa región -> lista de códigos de provincia (INEC).

    Returns:
        Copia de ``events`` con la columna ``region`` (``NaN`` si la
        provincia no está listada, p. ej. Galápagos, excluida por n=26).

    Raises:
        AsociacionENSOError: Si falta ``province_code``.
    """
    if "province_code" not in events.columns:
        raise AsociacionENSOError("La tabla de eventos no tiene 'province_code'.")
    mapa = {int(p): reg for reg, provs in regiones.items() for p in provs}
    out = events.copy()
    out["region"] = out["province_code"].astype(int).map(mapa)
    return out


def build_monthly_panel(events: pd.DataFrame, enso: pd.DataFrame) -> pd.DataFrame:
    """Construye el panel mensual completo (con ceros) para un subconjunto.

    Todos los meses del índice ENSO aparecen aunque no tengan eventos: omitir
    los meses con cero sesgaría la tasa al alza y violaría el principio de
    "nada se descarta silenciosamente" del proyecto.

    Args:
        events: Eventos del estrato (columnas ``year``, ``month``).
        enso: Índice ENSO mensual con ``year``, ``month`` y categorías.

    Returns:
        DataFrame con una fila por mes del índice ENSO, columna ``n`` (conteo)
        y ``t`` (tiempo estandarizado, media 0 y desviación 1).
    """
    conteo = events.groupby(["year", "month"]).size().rename("n").reset_index()
    panel = enso.merge(conteo, on=["year", "month"], how="left")
    panel["n"] = panel["n"].fillna(0).astype(int)
    tiempo = (panel["year"] - panel["year"].min()) + (panel["month"] - 1) / 12.0
    panel["t"] = (tiempo - tiempo.mean()) / tiempo.std()
    return panel.sort_values(["year", "month"]).reset_index(drop=True)


def detect_episodes(enso: pd.DataFrame, flag_col: str) -> list[tuple[int, int]]:
    """Detecta episodios como bloques contiguos de meses con ``flag_col`` True.

    Args:
        enso: Índice mensual ordenado cronológicamente.
        flag_col: Columna booleana que marca el mes como El Niño.

    Returns:
        Lista de tuplas (índice_inicio, índice_fin) inclusivas sobre ``enso``.
    """
    flags = enso[flag_col].astype(bool).to_numpy()
    episodios: list[tuple[int, int]] = []
    inicio: int | None = None
    for i, activo in enumerate(flags):
        if activo and inicio is None:
            inicio = i
        elif not activo and inicio is not None:
            episodios.append((inicio, i - 1))
            inicio = None
    if inicio is not None:
        episodios.append((inicio, len(flags) - 1))
    return episodios


# ---------------------------------------------------------------------------
# Matrices de diseño
# ---------------------------------------------------------------------------
def _seasonal_terms(panel: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Términos de estacionalidad: efectos fijos de mes o armónicos."""
    if kind == "month_fe":
        return pd.get_dummies(panel["month"], prefix="m", drop_first=True).astype(float)
    if kind == "harmonic":
        ang = 2 * np.pi * panel["month"] / 12.0
        return pd.DataFrame(
            {
                "sin1": np.sin(ang), "cos1": np.cos(ang),
                "sin2": np.sin(2 * ang), "cos2": np.cos(2 * ang),
            },
            index=panel.index,
        )
    raise AsociacionENSOError(f"Estacionalidad desconocida: {kind!r}")


def enso_terms(panel: pd.DataFrame, spec: str, flag_col: str, lag: int = 0) -> pd.DataFrame:
    """Construye los términos ENSO de interés para una especificación.

    Args:
        panel: Panel mensual con columnas ENSO.
        spec: ``"binary"`` (El Niño sí/no), ``"categorical"`` (La Niña, Débil,
            Moderado, Fuerte vs. Neutro), ``"ordinal"`` (El Niño + pendiente
            por categoría), o ``"continuous"`` (ICEN).
        flag_col: Columna booleana de mes El Niño para ``binary``/``ordinal``.
        lag: Rezago en meses (el efecto de ENSO en t-lag sobre eventos en t).

    Returns:
        DataFrame con los términos ENSO (sin constante).
    """
    cat = panel["icen_category"].astype(str)
    en = panel[flag_col].astype(float)
    if spec == "binary":
        terms = pd.DataFrame({"ElNino": en})
    elif spec == "categorical":
        terms = pd.DataFrame(
            {
                "LaNina": cat.str.startswith("La Ni").astype(float),
                "ElNino_Debil": (cat == "El Niño - Débil").astype(float),
                "ElNino_Moderado": (cat == "El Niño - Moderado").astype(float),
                "ElNino_Fuerte": (cat == "El Niño - Fuerte").astype(float),
            }
        )
    elif spec == "ordinal":
        score = cat.map({c: i for i, c in enumerate(ICEN_CATEGORIES_EL_NINO)}).fillna(0)
        terms = pd.DataFrame({"ElNino": en, "pendiente_categoria": score.astype(float)})
    elif spec == "continuous":
        terms = pd.DataFrame({"ICEN": panel["ICEN"].astype(float)})
    else:
        raise AsociacionENSOError(f"Especificación desconocida: {spec!r}")
    return terms.shift(lag) if lag else terms


def design_matrix(
    panel: pd.DataFrame,
    spec: str,
    flag_col: str,
    seasonality: str = "month_fe",
    lag: int = 0,
    year_fe: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Matriz de diseño y respuesta, eliminando filas sin dato por el rezago.

    Args:
        panel: Panel mensual.
        spec: Especificación ENSO (ver :func:`enso_terms`).
        flag_col: Columna booleana de mes El Niño.
        seasonality: ``"month_fe"`` o ``"harmonic"``.
        lag: Rezago ENSO en meses.
        year_fe: Si True, efectos fijos de año en vez de tendencia lineal.

    Returns:
        Tupla (X con constante, y con conteos), alineados.
    """
    partes = [enso_terms(panel, spec, flag_col, lag), _seasonal_terms(panel, seasonality)]
    if year_fe:
        partes.append(pd.get_dummies(panel["year"], prefix="y", drop_first=True).astype(float))
    else:
        partes.append(panel[["t"]])
    x = pd.concat(partes, axis=1)
    valido = x.notna().all(axis=1)
    x = sm.add_constant(x.loc[valido].astype(float), has_constant="add")
    return x, panel.loc[valido, "n"]


# ---------------------------------------------------------------------------
# Ajuste
# ---------------------------------------------------------------------------
def fit_negbin(y: pd.Series, x: pd.DataFrame, hac_lags: int = 3, robust: bool = True) -> FitResult:
    """Ajusta una regresión Binomial Negativa (NB2) con errores HAC.

    Inicializa con un Poisson GLM (estable) y optimiza el NB por BFGS.

    Args:
        y: Conteos mensuales.
        x: Matriz de diseño con constante.
        hac_lags: Rezagos de Newey-West para los errores estándar.
        robust: Si False, errores clásicos (necesarios para pruebas LR).

    Returns:
        :class:`FitResult` con los coeficientes y diagnósticos.

    Raises:
        AsociacionENSOError: Si el modelo no se puede ajustar.
    """
    try:
        poisson = sm.GLM(y, x, family=sm.families.Poisson()).fit()
        inicio = np.append(poisson.params.to_numpy(), 0.5)
        kwargs: dict[str, Any] = {}
        if robust:
            kwargs = {"cov_type": "HAC", "cov_kwds": {"maxlags": hac_lags}}
        modelo = sm.NegativeBinomial(y, x).fit(
            start_params=inicio, method="bfgs", maxiter=1000, disp=0, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 -- statsmodels lanza varios tipos
        raise AsociacionENSOError(f"No se pudo ajustar el NB: {exc}") from exc

    alpha = float(modelo.params["alpha"])
    mu = np.asarray(modelo.predict(x))
    varianza = mu + alpha * mu**2
    resid = (y.to_numpy() - mu) / np.sqrt(varianza)
    return FitResult(
        params=modelo.params.drop("alpha"),
        conf_int=modelo.conf_int().drop("alpha"),
        pvalues=modelo.pvalues.drop("alpha"),
        alpha=alpha,
        aic=float(modelo.aic),
        llf=float(modelo.llf),
        converged=bool(modelo.mle_retvals.get("converged", False)),
        resid_pearson=resid,
        n_obs=int(len(y)),
    )


def _irr_rows(fit: FitResult, terms: list[str], meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Convierte coeficientes log en razones de tasas (IRR) con IC 95 %."""
    filas = []
    for term in terms:
        filas.append(
            {
                **meta,
                "termino": term,
                "IRR": float(np.exp(fit.params[term])),
                "IC95_inf": float(np.exp(fit.conf_int.loc[term, 0])),
                "IC95_sup": float(np.exp(fit.conf_int.loc[term, 1])),
                "p_valor": float(fit.pvalues[term]),
                "alpha": fit.alpha,
                "convergio": fit.converged,
                "n_meses": fit.n_obs,
            }
        )
    return filas


def crude_rate_ratio(panel: pd.DataFrame, flag_col: str) -> float:
    """Razón cruda de eventos/mes (El Niño vs. resto), sin ajustar."""
    en = panel[flag_col].astype(bool)
    return float(panel.loc[en, "n"].mean() / panel.loc[~en, "n"].mean())


# ---------------------------------------------------------------------------
# Análisis por estrato
# ---------------------------------------------------------------------------
def analyze_stratum(
    events: pd.DataFrame, enso: pd.DataFrame, name: str, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Corre todas las especificaciones sobre un estrato (región/amenaza).

    Args:
        events: Eventos del estrato.
        enso: Índice ENSO mensual extendido.
        name: Nombre del estrato (para trazabilidad).
        cfg: Sección ``enso_association`` de ``config.yaml``.

    Returns:
        Diccionario de tablas: ``irr`` (todas las especificaciones),
        ``dosis`` (categorías y pendiente), ``rezagos`` y ``diag``.
    """
    flag = cfg["flag_icen"]
    flag_oni = cfg["flag_oni"]
    hac = cfg["hac_lags"]
    panel = build_monthly_panel(events, enso)
    n_ev = int(panel["n"].sum())
    meta = {"estrato": name, "n_eventos": n_ev}
    filas: list[dict[str, Any]] = []

    # Modelo primario y sensibilidades de la especificación binaria.
    variantes = {
        "M1_primario_ICEN_binario": dict(flag_col=flag, seasonality="month_fe", year_fe=False),
        "S1_ONI_binario": dict(flag_col=flag_oni, seasonality="month_fe", year_fe=False),
        "S2_estacionalidad_armonica": dict(flag_col=flag, seasonality="harmonic", year_fe=False),
        "S3_efectos_fijos_anio": dict(flag_col=flag, seasonality="month_fe", year_fe=True),
    }
    primario: FitResult | None = None
    for nombre, kw in variantes.items():
        x, y = design_matrix(panel, "binary", **kw)
        fit = fit_negbin(y, x, hac)
        if nombre.startswith("M1"):
            primario = fit
        filas += _irr_rows(
            fit, ["ElNino"], {**meta, "modelo": nombre, "IRR_crudo": crude_rate_ratio(panel, kw["flag_col"])}
        )

    # ICEN continuo (por +1 unidad de ICEN).
    x, y = design_matrix(panel, "continuous", flag)
    filas += _irr_rows(fit_negbin(y, x, hac), ["ICEN"], {**meta, "modelo": "S4_ICEN_continuo", "IRR_crudo": np.nan})

    # Dosis-respuesta: categorías y pendiente ordinal.
    x, y = design_matrix(panel, "categorical", flag)
    fit_cat = fit_negbin(y, x, hac)
    dosis = pd.DataFrame(
        _irr_rows(fit_cat, ["LaNina", "ElNino_Debil", "ElNino_Moderado", "ElNino_Fuerte"], {**meta, "modelo": "M2_categorico"})
    )
    x, y = design_matrix(panel, "ordinal", flag)
    fit_ord = fit_negbin(y, x, hac)
    dosis = pd.concat(
        [dosis, pd.DataFrame(_irr_rows(fit_ord, ["pendiente_categoria"], {**meta, "modelo": "M2b_ordinal"}))],
        ignore_index=True,
    )

    # Rezagos 0-3.
    rez = []
    for lag in range(0, cfg["max_lag"] + 1):
        x, y = design_matrix(panel, "binary", flag, lag=lag)
        f = fit_negbin(y, x, hac)
        rez += _irr_rows(f, ["ElNino"], {**meta, "modelo": f"rezago_{lag}", "rezago": lag, "aic": f.aic})

    # Diagnósticos del modelo primario.
    assert primario is not None  # M1 siempre se ajusta arriba
    lb = acorr_ljungbox(primario.resid_pearson, lags=[12], return_df=True)
    diag = pd.DataFrame(
        [{
            **meta,
            "alpha_dispersion": primario.alpha,
            "ljung_box_p_lag12": float(lb["lb_pvalue"].iloc[0]),
            "convergio": primario.converged,
            "aic": primario.aic,
            "n_meses_enso": int(panel[flag].sum()),
        }]
    )
    return {"irr": pd.DataFrame(filas), "dosis": dosis, "rezagos": pd.DataFrame(rez), "diag": diag}


def leave_one_episode_out(
    events: pd.DataFrame, enso: pd.DataFrame, name: str, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Reajusta el modelo primario excluyendo, uno a uno, cada episodio.

    Con ~6 episodios independientes, un solo episodio (típicamente 2015-16)
    puede dominar el efecto; esta es la verificación más honesta de ello.

    Args:
        events: Eventos del estrato.
        enso: Índice ENSO mensual.
        name: Nombre del estrato.
        cfg: Configuración de la etapa.

    Returns:
        Tabla con el IRR obtenido al excluir cada episodio.
    """
    panel_full = build_monthly_panel(events, enso)
    episodios = detect_episodes(panel_full, cfg["flag_icen"])
    filas = []
    for k, (i0, i1) in enumerate(episodios, start=1):
        sub = panel_full.drop(index=range(i0, i1 + 1)).reset_index(drop=True)
        x, y = design_matrix(sub, "binary", cfg["flag_icen"])
        fit = fit_negbin(y, x, cfg["hac_lags"])
        periodo = f"{int(panel_full.loc[i0, 'year'])}-{int(panel_full.loc[i0, 'month']):02d} a " \
                  f"{int(panel_full.loc[i1, 'year'])}-{int(panel_full.loc[i1, 'month']):02d}"
        filas += _irr_rows(
            fit, ["ElNino"],
            {"estrato": name, "modelo": "LOEO", "episodio_excluido": k, "periodo": periodo, "n_eventos": int(panel_full["n"].sum())},
        )
    return pd.DataFrame(filas)


def block_bootstrap_irr(
    events: pd.DataFrame, enso: pd.DataFrame, name: str, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Intervalo de bootstrap por bloques móviles para el IRR primario.

    Remuestrea bloques consecutivos de meses para preservar la autocorrelación.
    Es una verificación independiente de los errores HAC.

    Args:
        events: Eventos del estrato.
        enso: Índice ENSO mensual.
        name: Nombre del estrato.
        cfg: Configuración (``bootstrap_B``, ``bootstrap_block``, ``seed``).

    Returns:
        Diccionario con IRR e IC 95 % percentil del bootstrap.
    """
    panel = build_monthly_panel(events, enso)
    x, y = design_matrix(panel, "binary", cfg["flag_icen"])
    n, b, largo = len(y), cfg["bootstrap_B"], cfg["bootstrap_block"]
    rng = np.random.default_rng(cfg["seed"])
    estimaciones: list[float] = []
    n_bloques = int(np.ceil(n / largo))
    for _ in range(b):
        inicios = rng.integers(0, n - largo + 1, size=n_bloques)
        idx = np.concatenate([np.arange(s, s + largo) for s in inicios])[:n]
        xb, yb = x.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
        if xb["ElNino"].nunique() < 2 or np.linalg.matrix_rank(xb.to_numpy()) < xb.shape[1]:
            continue  # remuestra sin variación en ENSO o rango deficiente: se omite
        try:
            f = fit_negbin(yb, xb, cfg["hac_lags"], robust=False)
        except AsociacionENSOError:
            continue
        estimaciones.append(float(f.params["ElNino"]))
    if len(estimaciones) < 0.8 * b:
        logger.warning("Bootstrap %s: solo %d/%d réplicas válidas.", name, len(estimaciones), b)
    arr = np.exp(np.array(estimaciones))
    return {
        "estrato": name,
        "boot_IRR_mediana": float(np.median(arr)),
        "boot_IC95_inf": float(np.percentile(arr, 2.5)),
        "boot_IC95_sup": float(np.percentile(arr, 97.5)),
        "boot_replicas_validas": int(len(arr)),
    }


def heterogeneity_test(
    events: pd.DataFrame, enso: pd.DataFrame, regiones: list[str], cfg: dict[str, Any]
) -> pd.DataFrame:
    """Prueba de razón de verosimilitud: ¿difiere el efecto ENSO entre regiones?

    Compara un modelo con un único efecto ENSO común contra otro con efecto
    por región, ambos con estacionalidad y tendencia propias de cada región.
    Se usan errores clásicos (no HAC) porque la prueba LR lo exige.

    Args:
        events: Eventos con columna ``region``.
        enso: Índice ENSO mensual.
        regiones: Regiones a incluir.
        cfg: Configuración de la etapa.

    Returns:
        Tabla de una fila con estadístico chi-cuadrado, g.l. y p-valor.
    """
    bloques_x, bloques_y = [], []
    for reg in regiones:
        panel = build_monthly_panel(events[events["region"] == reg], enso)
        x, y = design_matrix(panel, "binary", cfg["flag_icen"])
        x = x.drop(columns="ElNino")
        x.columns = [f"{reg}_{c}" for c in x.columns]
        x[f"{reg}_ElNino"] = panel[cfg["flag_icen"]].astype(float).to_numpy()
        bloques_x.append(x)
        bloques_y.append(y)
    x_all = pd.concat(bloques_x, axis=0).fillna(0.0).reset_index(drop=True)
    y_all = pd.concat(bloques_y, axis=0).reset_index(drop=True)
    cols_en = [f"{r}_ElNino" for r in regiones]
    x_comun = x_all.drop(columns=cols_en).copy()
    x_comun["ElNino_comun"] = x_all[cols_en].sum(axis=1)

    def _llf(xm: pd.DataFrame) -> float:
        p = sm.GLM(y_all, xm, family=sm.families.Poisson()).fit()
        m = sm.NegativeBinomial(y_all, xm).fit(
            start_params=np.append(p.params.to_numpy(), 0.5), method="bfgs", maxiter=1000, disp=0
        )
        return float(m.llf)

    lr = 2 * (_llf(x_all) - _llf(x_comun))
    gl = len(regiones) - 1
    return pd.DataFrame(
        [{"prueba": "LR efecto ENSO por región vs. común", "chi2": lr, "gl": gl, "p_valor": float(stats.chi2.sf(lr, gl))}]
    )


def apply_fdr(irr: pd.DataFrame, modelo: str, alpha: float = 0.05) -> pd.DataFrame:
    """Aplica corrección Benjamini-Hochberg a las pruebas del modelo primario.

    Args:
        irr: Tabla de IRR de todos los estratos y modelos.
        modelo: Nombre del modelo cuya familia de pruebas se corrige.
        alpha: Tasa de falsos descubrimientos objetivo.

    Returns:
        ``irr`` con columnas ``p_BH`` y ``significativo_BH`` (solo para ese modelo).
    """
    out = irr.copy()
    out["p_BH"] = np.nan
    out["significativo_BH"] = pd.NA
    mask = out["modelo"] == modelo
    rechazo, p_adj, _, _ = multipletests(out.loc[mask, "p_valor"], alpha=alpha, method="fdr_bh")
    out.loc[mask, "p_BH"] = p_adj
    out.loc[mask, "significativo_BH"] = rechazo
    return out


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run_analysis(
    events: pd.DataFrame, enso: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Ejecuta la etapa de asociación ENSO–amenaza completa sobre todos los estratos definidos.

    Args:
        events: Tabla ``hazard_event`` con ``hazard_type`` y ``province_code``.
        enso: ``enso_index_monthly_extended``.
        cfg: Sección ``enso_association`` de ``config.yaml``.

    Returns:
        Diccionario de tablas de resultados (irr, dosis, rezagos, diag, loeo,
        bootstrap, heterogeneidad).

    Raises:
        AsociacionENSOError: Si faltan columnas obligatorias.
    """
    for col in ("hazard_type", "province_code", "year", "month"):
        if col not in events.columns:
            raise AsociacionENSOError(f"Falta la columna '{col}' en los eventos.")
    ev = assign_region(events, cfg["regiones"])
    estratos: dict[str, pd.DataFrame] = {"Nacional": ev}
    for reg in cfg["regiones"]:
        estratos[f"Región {reg}"] = ev[ev["region"] == reg]
    for haz in sorted(ev["hazard_type"].unique()):
        estratos[f"Amenaza {haz}"] = ev[ev["hazard_type"] == haz]
    for reg in cfg["regiones"]:
        for haz in sorted(ev["hazard_type"].unique()):
            sub = ev[(ev["region"] == reg) & (ev["hazard_type"] == haz)]
            if len(sub) >= cfg["min_eventos_estrato"]:
                estratos[f"{reg} × {haz}"] = sub
            else:
                logger.info("Estrato %s × %s omitido: %d eventos < mínimo %d.", reg, haz, len(sub), cfg["min_eventos_estrato"])

    tablas: dict[str, list[pd.DataFrame]] = {k: [] for k in ("irr", "dosis", "rezagos", "diag", "loeo", "bootstrap")}
    primarios = ["Nacional"] + [f"Región {r}" for r in cfg["regiones"]] + [
        f"Amenaza {h}" for h in sorted(ev["hazard_type"].unique())
    ]
    for nombre, sub in estratos.items():
        logger.info("Estrato %s: %d eventos", nombre, len(sub))
        res = analyze_stratum(sub, enso, nombre, cfg)
        for k, v in res.items():
            tablas[k].append(v)
        tablas["loeo"].append(leave_one_episode_out(sub, enso, nombre, cfg))
        if nombre in primarios:
            tablas["bootstrap"].append(pd.DataFrame([block_bootstrap_irr(sub, enso, nombre, cfg)]))

    salida = {k: pd.concat(v, ignore_index=True) for k, v in tablas.items()}
    salida["irr"] = apply_fdr(salida["irr"], "M1_primario_ICEN_binario", cfg["fdr_alpha"])
    salida["heterogeneidad"] = heterogeneity_test(ev, enso, list(cfg["regiones"]), cfg)
    return salida


def export_tables(tablas: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    """Exporta las tablas de resultados en Parquet y CSV.

    Args:
        tablas: Resultado de :func:`run_analysis`.
        out_dir: Directorio de salida (``data/processed``).

    Returns:
        Lista de rutas escritas.
    """
    destino = Path(out_dir)
    destino.mkdir(parents=True, exist_ok=True)
    escritas: list[Path] = []
    for nombre, df in tablas.items():
        base = destino / f"enso_asociacion_{nombre}"
        df.to_parquet(base.with_suffix(".parquet"), index=False)
        df.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8")
        escritas += [base.with_suffix(".parquet"), base.with_suffix(".csv")]
    logger.info("Exportadas %d tablas a %s", len(tablas), destino)
    return escritas


# ---------------------------------------------------------------------------
# Etiqueta de señal ENSO por cantón (insumo del dashboard y los perfiles)
# ---------------------------------------------------------------------------
def classify_signal(
    irr: float, significativo: bool, umbral_fuerte: float = 2.0, umbral_moderada: float = 1.2
) -> str:
    """Clasifica la señal ENSO de un estrato a partir de su IRR y significancia.

    Args:
        irr: Razón de tasas ajustada (El Niño vs. resto).
        significativo: Si la prueba resulta significativa tras corrección BH.
        umbral_fuerte: IRR mínimo para "Fuerte".
        umbral_moderada: IRR mínimo para "Moderada".

    Returns:
        Una de: ``"Fuerte"``, ``"Moderada"``, ``"Débil"``, ``"Inversa"`` o
        ``"No detectable"``.
    """
    if not significativo or pd.isna(irr):
        return "No detectable"
    if irr < 1.0:
        return "Inversa"
    if irr >= umbral_fuerte:
        return "Fuerte"
    if irr >= umbral_moderada:
        return "Moderada"
    return "Débil"


def build_signal_table(
    efci: pd.DataFrame, irr: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Asigna a cada cantón × amenaza la etiqueta de señal ENSO de su región.

    La etiqueta de ``TODOS`` usa el estrato regional; la de cada amenaza usa el
    estrato región × amenaza si existe (≥ ``min_eventos_estrato`` eventos) y,
    si no, queda "No evaluado" (pocos eventos). Galápagos queda "No evaluado".

    Args:
        efci: ``efci_composite_canton`` (``canton_code``, ``province_code``,
            ``hazard_type``).
        irr: Tabla ``irr`` de :func:`run_analysis` (con ``p_BH``).
        cfg: Configuración de la etapa.

    Returns:
        Tabla cantón × amenaza con región, IRR, IC 95 % y etiqueta.
    """
    mapa = {int(p): r for r, provs in cfg["regiones"].items() for p in provs}
    prim = irr[irr["modelo"] == "M1_primario_ICEN_binario"].set_index("estrato")
    filas = []
    for _, fila in efci[
        ["canton_code", "canton_name_official", "province_code", "hazard_type"]
    ].iterrows():
        region = mapa.get(int(fila["province_code"]))
        haz = fila["hazard_type"]
        estrato = None
        if region is not None:
            estrato = f"Región {region}" if haz == "TODOS" else f"{region} × {haz}"
        if estrato is not None and estrato in prim.index:
            r = prim.loc[estrato]
            etiqueta = classify_signal(
                r["IRR"], bool(r["significativo_BH"]),
                cfg["senal_umbral_fuerte"], cfg["senal_umbral_moderada"],
            )
            datos = (float(r["IRR"]), float(r["IC95_inf"]), float(r["IC95_sup"]))
        else:
            etiqueta, datos = "No evaluado", (np.nan, np.nan, np.nan)
        filas.append(
            {
                "canton_code": fila["canton_code"],
                "canton_name_official": fila["canton_name_official"],
                "hazard_type": haz,
                "region": region if region is not None else "Galápagos",
                "estrato_senal": estrato,
                "IRR": datos[0], "IC95_inf": datos[1], "IC95_sup": datos[2],
                "senal_enso": etiqueta,
            }
        )
    return pd.DataFrame(filas)
