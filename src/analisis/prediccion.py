"""Componente predictivo del EFCI (etapa de componente predictivo, etapa de diseño conceptual §10).

Estima el impacto esperado (población, agricultura, vivienda) y la
probabilidad de severidad alta para cada cantón ante un nuevo episodio
El Niño, entrenando sobre los episodios históricos 2015-2025.

Decisión de diseño central (confirmada por Jerson vía AskUserQuestion,
2026-08-06): la unidad de predicción es **cantón × episodio ENSO**, no
cantón × mes — un episodio El Niño completo es la unidad de decisión real
("si viene un Niño fuerte, ¿qué esperar en el cantón X?"), y agregar por
episodio evita inflar artificialmente el tamaño muestral con meses que en
el fondo describen el mismo evento climático.

Con solo 5-6 episodios históricos independientes, cualquier algoritmo de
ML tiene poder predictivo limitado — no es un problema de elegir mejor
algoritmo, es un problema de tamaño de muestra. Por eso este módulo separa
explícitamente el modelo de un *baseline* ingenuo (Sección "Baseline") y
reporta ambos lado a lado en vez de solo el modelo, siguiendo la
instrucción explícita de Jerson de manejar N=5-6 con "modelo simple +
baseline de referencia" en vez de forzar un resultado optimista.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Definición de episodios ENSO
# -----------------------------------------------------------------------
# Redefinidos aquí a partir del histórico real de 38 meses "El Niño" en
# enso_index_monthly_extended.parquet, agrupando por bloques de
# meses CONTIGUOS -- no por año calendario. Esto es una desviación
# deliberada y documentada de la lista de "5 episodios" usada en la etapa de susceptibilidad SAR
# para seleccionar fechas SAR (2015, 2016, 2017, 2023, 2024): aquella
# lista trataba 2015 y 2016 como dos episodios separados por conveniencia
# de muestreo (3 fechas por año calendario), pero climatológicamente
# 2015-04 a 2016-06 es UN solo evento El Niño continuo (el "El Niño
# 2015-16"), y partirlo en dos introduciría autocorrelación falsa entre
# folds de validación. Verificado programáticamente (no supuesto): la
# unión de estos 6 bloques reproduce EXACTAMENTE los 38 meses "El Niño"
# del índice ICEN, y los 6 bloques tienen entre 528 y 2,394 eventos de
# peligro reales cada uno (ninguno es residual/vacío) -- ver notebook
# Step12, celda de verificación.
EPISODIOS_ENSO: dict[str, list[tuple[int, int]]] = {
    "2015-2016": [(2015, m) for m in range(4, 13)] + [(2016, m) for m in range(1, 7)],
    "2017": [(2017, m) for m in range(1, 5)],
    "2018-2019": [(2018, 11), (2018, 12), (2019, 1), (2019, 2)],
    "2023": [(2023, m) for m in range(3, 13)],
    "2024": [(2024, 1), (2024, 2), (2024, 3)],
    "2025": [(2025, 3), (2025, 4)],
}

# Traducción ordinal de categoría ICEN, reutilizando exactamente la escala
# ya validada y documentada en la etapa de índice de magnitud ENSO (enso_magnitude_index) -- no se
# reinventa una escala nueva.
MAGNITUD_ORDINAL: dict[str, float] = {
    "Neutro": 0.0,
    "La Niña": 0.0,
    "El Niño - Débil": 0.25,
    "El Niño - Moderado": 0.50,
    "El Niño - Fuerte": 0.75,
    "El Niño - Extraordinario": 1.00,
}

TARGET_COLS: list[str] = ["poblacion_afectada", "agricultura_perdida_ha", "vivienda_afectada"]


def meses_de_episodio(nombre_episodio: str) -> list[tuple[int, int]]:
    """Devuelve la lista (año, mes) de un episodio ENSO definido en EPISODIOS_ENSO.

    Args:
        nombre_episodio: Clave de ``EPISODIOS_ENSO`` (p. ej. ``"2023"``).

    Returns:
        Lista de tuplas (año, mes) que componen el episodio.
    """
    return EPISODIOS_ENSO[nombre_episodio]


def construir_panel_canton_episodio(
    hazard_event: pd.DataFrame,
    enso_monthly: pd.DataFrame,
    universo_cantones: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el panel cantón × episodio con targets de impacto y magnitud ENSO.

    Args:
        hazard_event: Tabla de eventos (``data/processed/hazard_event.parquet``),
            debe incluir ``canton_code``, ``year``, ``month`` y las columnas
            de impacto crudo.
        enso_monthly: Serie ENSO mensual extendida, con
            ``icen_category`` por (año, mes).
        universo_cantones: Tabla con una fila por cada uno de los 221
            cantones (``canton_code``), para garantizar cobertura completa
            incluso de cantones sin eventos en un episodio dado (targets=0,
            no filas ausentes -- mismo patrón que etapa de índice de recurrencia §"bug corregido").

    Returns:
        DataFrame con una fila por (canton_code, episodio): n_eventos,
        los 3 targets de impacto crudo, y magnitud_enso_media/max del
        episodio (misma para todos los cantones de ese episodio, es una
        característica climática, no espacial).
    """
    filas: list[dict[str, Any]] = []
    for episodio, meses in EPISODIOS_ENSO.items():
        meses_set = set(meses)
        en_episodio = hazard_event.apply(
            lambda r: (int(r["year"]), int(r["month"])) in meses_set, axis=1
        )
        sub = hazard_event.loc[en_episodio]

        # Magnitud ENSO del episodio: promedio y máximo ordinal de sus meses,
        # a partir de la categoría ICEN real (no supuesta) por mes.
        cats = enso_monthly.set_index(["year", "month"]).loc[list(meses_set), "icen_category"]
        ords = cats.map(MAGNITUD_ORDINAL).fillna(0.0)
        mag_media = float(ords.mean())
        mag_max = float(ords.max())

        agg = (
            sub.groupby("canton_code")
            .agg(
                n_eventos=("event_id", "count"),
                poblacion_afectada=("affected_direct", "sum"),
                poblacion_afectada_indirecta=("affected_indirect", "sum"),
                agricultura_afectada_ha=("crops_affected_ha", "sum"),
                agricultura_destruida_ha=("crops_destroyed_ha", "sum"),
                vivienda_afectada_n=("houses_affected", "sum"),
                vivienda_destruida_n=("houses_destroyed", "sum"),
            )
            .reset_index()
        )
        agg["poblacion_afectada"] = agg["poblacion_afectada"] + agg.pop("poblacion_afectada_indirecta")
        agg["agricultura_perdida_ha"] = agg.pop("agricultura_afectada_ha") + agg.pop("agricultura_destruida_ha")
        agg["vivienda_afectada"] = agg.pop("vivienda_afectada_n") + agg.pop("vivienda_destruida_n")

        # Merge contra el universo completo de 221 cantones: los que no
        # tuvieron ningún evento este episodio quedan en 0, no ausentes.
        full = universo_cantones[["canton_code"]].merge(agg, on="canton_code", how="left")
        full[["n_eventos", "poblacion_afectada", "agricultura_perdida_ha", "vivienda_afectada"]] = full[
            ["n_eventos", "poblacion_afectada", "agricultura_perdida_ha", "vivienda_afectada"]
        ].fillna(0)
        full["episodio"] = episodio
        full["magnitud_enso_media"] = mag_media
        full["magnitud_enso_max"] = mag_max
        filas.append(full)

    panel = pd.concat(filas, ignore_index=True)
    logger.info(
        "Panel cantón×episodio construido: %d filas (%d cantones × %d episodios)",
        len(panel),
        panel["canton_code"].nunique(),
        panel["episodio"].nunique(),
    )
    return panel


def construir_features_sin_fuga(panel: pd.DataFrame, episodio_objetivo: str) -> pd.DataFrame:
    """Calcula features históricas por cantón usando SOLO episodios de entrenamiento.

    Este es el paso que evita fuga de información (leakage) en la
    validación leave-one-episode-out: para el fold donde ``episodio_objetivo``
    es el conjunto de prueba, ninguna feature puede derivarse de datos de
    ese mismo episodio -- solo de los otros episodios (ventana expansiva,
    no toda la serie).

    Args:
        panel: Panel completo cantón×episodio (salida de
            :func:`construir_panel_canton_episodio`).
        episodio_objetivo: Nombre del episodio que se deja fuera del
            entrenamiento (el que se va a predecir en este fold).

    Returns:
        DataFrame indexado por ``canton_code`` con las features históricas
        (sin ninguna columna del episodio objetivo).
    """
    entrenamiento = panel[panel["episodio"] != episodio_objetivo]

    hist = (
        entrenamiento.groupby("canton_code")
        .agg(
            hist_n_episodios_con_evento=("n_eventos", lambda s: int((s > 0).sum())),
            hist_poblacion_media=("poblacion_afectada", "mean"),
            hist_agricultura_media=("agricultura_perdida_ha", "mean"),
            hist_vivienda_media=("vivienda_afectada", "mean"),
            hist_n_eventos_medio=("n_eventos", "mean"),
        )
        .reset_index()
    )
    hist["hist_frac_episodios_con_evento"] = hist["hist_n_episodios_con_evento"] / entrenamiento[
        "episodio"
    ].nunique()
    return hist


def severidad_binaria(panel: pd.DataFrame, umbral_percentil: float = 0.80) -> pd.Series:
    """Define la etiqueta binaria 'impacto severo' a partir de un score compuesto.

    Score = suma de los 3 targets normalizados 0-1 por min-max GLOBAL
    (mismo criterio de normalización que etapa de índice de severidad/8, para mantener
    consistencia metodológica dentro del proyecto). El umbral (percentil 80
    por defecto, entre filas con score>0) define la clase positiva --
    documentado como una elección de diseño, no un valor mágico.

    Args:
        panel: Panel completo cantón×episodio con los 3 targets crudos.
        umbral_percentil: Percentil (dentro de las filas con score>0) sobre
            el cual una fila se etiqueta como severa.

    Returns:
        Serie booleana alineada con el índice de ``panel``.
    """
    score = pd.DataFrame(index=panel.index)
    for col in TARGET_COLS:
        vmax = panel[col].max()
        score[col] = panel[col] / vmax if vmax > 0 else 0.0
    total = score.sum(axis=1)
    positivos = total[total > 0]
    umbral = positivos.quantile(umbral_percentil) if len(positivos) else 0.0
    return total >= umbral if umbral > 0 else pd.Series(False, index=panel.index)


def evaluar_loeo(
    panel: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 300,
) -> dict[str, Any]:
    """Corre la validación leave-one-episode-out completa (modelo + baseline).

    Para cada episodio E de los 6 disponibles: entrena con los otros 5,
    predice E, y compara contra un baseline ingenuo (la media histórica del
    propio cantón en los episodios de entrenamiento -- "lo que ya sabíamos
    de este cantón, sin ningún modelo"). Reporta métricas fold por fold y
    agregadas, para los 3 targets de regresión y la clasificación binaria
    de severidad.

    Args:
        panel: Panel completo cantón×episodio.
        random_state: Semilla para reproducibilidad del Random Forest.
        n_estimators: Número de árboles del Random Forest (regresión y
            clasificación).

    Returns:
        Diccionario con: ``metricas_fold`` (DataFrame largo, una fila por
        fold×target×métrica×modelo), ``predicciones`` (DataFrame con
        predicciones e etiquetas reales por fila de prueba), y
        ``importancia_features`` (DataFrame de importancia de features del
        modelo final entrenado con los 6 episodios).
    """
    panel = panel.copy()
    panel["severo"] = severidad_binaria(panel)

    filas_metricas: list[dict[str, Any]] = []
    filas_pred: list[dict[str, Any]] = []
    feature_cols_base = [
        "magnitud_enso_media",
        "magnitud_enso_max",
        "hist_n_episodios_con_evento",
        "hist_frac_episodios_con_evento",
        "hist_poblacion_media",
        "hist_agricultura_media",
        "hist_vivienda_media",
        "hist_n_eventos_medio",
    ]

    for episodio in EPISODIOS_ENSO:
        hist = construir_features_sin_fuga(panel, episodio)
        train_mask = panel["episodio"] != episodio
        test_mask = panel["episodio"] == episodio

        train = panel[train_mask].merge(hist, on="canton_code", how="left")
        test = panel[test_mask].merge(hist, on="canton_code", how="left")
        # Cantones sin ningún episodio de entrenamiento con evento (NaN tras
        # el merge, no debería ocurrir dado que hist ya cubre el universo de
        # entrenamiento, pero se blinda igual): imputar a 0, es información
        # legítima ("este cantón nunca tuvo un evento en los episodios
        # previos"), no un dato faltante real.
        train[feature_cols_base] = train[feature_cols_base].fillna(0.0)
        test[feature_cols_base] = test[feature_cols_base].fillna(0.0)

        X_train, X_test = train[feature_cols_base], test[feature_cols_base]

        # --- Regresión (3 targets) ---
        for col in TARGET_COLS:
            y_train_log = np.log1p(train[col])
            y_test_real = test[col].to_numpy()

            modelo = RandomForestRegressor(
                n_estimators=n_estimators, random_state=random_state, min_samples_leaf=3
            )
            modelo.fit(X_train, y_train_log)
            pred_log = modelo.predict(X_test)
            pred_real = np.expm1(pred_log)
            pred_real = np.clip(pred_real, 0, None)

            # Baseline ingenuo: la media histórica del propio cantón (ya
            # está en hist_<col>_media, sin necesidad de recalcular).
            baseline_col = {
                "poblacion_afectada": "hist_poblacion_media",
                "agricultura_perdida_ha": "hist_agricultura_media",
                "vivienda_afectada": "hist_vivienda_media",
            }[col]
            pred_baseline = test[baseline_col].to_numpy()

            mae_modelo = mean_absolute_error(y_test_real, pred_real)
            mae_baseline = mean_absolute_error(y_test_real, pred_baseline)
            r2_modelo = r2_score(y_test_real, pred_real)
            r2_baseline = r2_score(y_test_real, pred_baseline)

            filas_metricas.append(
                dict(episodio=episodio, target=col, metrica="MAE", modelo=mae_modelo, baseline=mae_baseline)
            )
            filas_metricas.append(
                dict(episodio=episodio, target=col, metrica="R2", modelo=r2_modelo, baseline=r2_baseline)
            )

            for cc, real, pr_m, pr_b in zip(test["canton_code"], y_test_real, pred_real, pred_baseline):
                filas_pred.append(
                    dict(
                        episodio=episodio,
                        canton_code=cc,
                        target=col,
                        valor_real=real,
                        prediccion_modelo=pr_m,
                        prediccion_baseline=pr_b,
                    )
                )

        # --- Clasificación (severo sí/no) ---
        y_train_cls = train["severo"].astype(int)
        y_test_cls = test["severo"].astype(int)

        if y_train_cls.nunique() < 2:
            # Puede pasar en folds pequeños si ningún caso de entrenamiento
            # quedó marcado severo -- se documenta explícitamente en vez de
            # fallar silenciosamente.
            logger.warning(
                "Fold %s: solo una clase presente en entrenamiento, se omite clasificación.", episodio
            )
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, min_samples_leaf=3, class_weight="balanced"
        )
        clf.fit(X_train, y_train_cls)
        proba = clf.predict_proba(X_test)[:, 1]
        pred_cls = (proba >= 0.5).astype(int)

        # Baseline: frecuencia histórica de severidad del propio cantón
        # (fracción de episodios de entrenamiento donde tuvo algún evento,
        # como proxy simple de "cuán seguido es severo").
        baseline_proba = test["hist_frac_episodios_con_evento"].to_numpy()
        baseline_pred = (baseline_proba >= 0.5).astype(int)

        if y_test_cls.nunique() > 1:
            auc_modelo = roc_auc_score(y_test_cls, proba)
            auc_baseline = roc_auc_score(y_test_cls, baseline_proba)
        else:
            auc_modelo = np.nan
            auc_baseline = np.nan
        f1_modelo = f1_score(y_test_cls, pred_cls, zero_division=0)
        f1_baseline = f1_score(y_test_cls, baseline_pred, zero_division=0)

        filas_metricas.append(
            dict(episodio=episodio, target="severo", metrica="AUC", modelo=auc_modelo, baseline=auc_baseline)
        )
        filas_metricas.append(
            dict(episodio=episodio, target="severo", metrica="F1", modelo=f1_modelo, baseline=f1_baseline)
        )

    metricas = pd.DataFrame(filas_metricas)
    predicciones = pd.DataFrame(filas_pred)

    # Modelo final "operativo": entrenado con los 6 episodios completos, no
    # validado (para eso está el LOEO de arriba) -- solo para producir
    # importancia de features y quedar listo para el próximo episodio real.
    hist_final = construir_features_sin_fuga(
        panel.assign(episodio="__ninguno__"), "__no_excluir_nada__"
    )
    # truco: al pasar un nombre de episodio que no existe en panel, la
    # función de arriba usa TODO el panel como "entrenamiento" -- reutiliza
    # la misma lógica sin duplicar código.
    panel_final = panel.merge(hist_final, on="canton_code", how="left")
    panel_final[feature_cols_base] = panel_final[feature_cols_base].fillna(0.0)
    importancias = {}
    for col in TARGET_COLS:
        modelo_final = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, min_samples_leaf=3)
        modelo_final.fit(panel_final[feature_cols_base], np.log1p(panel_final[col]))
        importancias[col] = modelo_final.feature_importances_
    importancia_df = pd.DataFrame(importancias, index=feature_cols_base).reset_index(names="feature")

    return dict(metricas_fold=metricas, predicciones=predicciones, importancia_features=importancia_df)
