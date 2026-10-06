"""Helpers de entrada/salida reutilizables (lectura de Excel, escritura de
Parquet/CSV) usados por los notebooks de todas las fases.

Mantener este módulo libre de lógica de negocio: solo E/S genérica.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_excel_sheet(workbook_path: str | Path, sheet_name: str, **kwargs) -> pd.DataFrame:
    """Lee una hoja de un workbook Excel con manejo de errores explícito.

    Args:
        workbook_path: Ruta al archivo ``.xlsx``.
        sheet_name: Nombre exacto de la hoja a leer.
        **kwargs: Argumentos adicionales reenviados a :func:`pandas.read_excel`
            (p. ej. ``header=``, ``dtype=``).

    Returns:
        DataFrame con el contenido de la hoja.

    Raises:
        FileNotFoundError: Si ``workbook_path`` no existe.
        ValueError: Si la hoja no existe dentro del workbook.
    """
    path = Path(workbook_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el workbook: {path}")

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", **kwargs)
    except ValueError as exc:
        # pandas lanza ValueError cuando la hoja no existe — lo re-envolvemos
        # con contexto (nombre de archivo) para que el error sea accionable.
        raise ValueError(f"No se pudo leer la hoja '{sheet_name}' de {path}: {exc}") from exc

    logger.info("Hoja '%s' leída desde %s -> %d filas x %d columnas", sheet_name, path.name, *df.shape)
    return df


def ensure_dir(path: str | Path) -> Path:
    """Crea un directorio (y sus padres) si no existe, y devuelve la ruta resuelta.

    Args:
        path: Ruta del directorio a asegurar.

    Returns:
        Ruta absoluta del directorio, ya garantizado que existe.
    """
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_processed_table(df: pd.DataFrame, output_dir: str | Path, base_name: str) -> dict[str, Path]:
    """Escribe un DataFrame procesado en Parquet y CSV dentro de ``output_dir``.

    Se escriben ambos formatos deliberadamente: Parquet para consumo eficiente
    por el resto del pipeline (tipado preservado, lectura rápida) y CSV para
    inspección manual rápida (Excel, revisión humana) — ver documento técnico
    de la etapa de ingesta de datos, sección "Formatos de salida".

    Args:
        df: Tabla ya limpia/armonizada a exportar.
        output_dir: Carpeta de destino (se crea si no existe).
        base_name: Nombre base del archivo, sin extensión (p. ej. ``"hazard_event"``).

    Returns:
        Diccionario ``{"parquet": ruta, "csv": ruta}`` con las rutas escritas.
    """
    out_dir = ensure_dir(output_dir)
    parquet_path = out_dir / f"{base_name}.parquet"
    csv_path = out_dir / f"{base_name}.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info("Tabla '%s' exportada: %s (%d filas)", base_name, parquet_path, len(df))
    return {"parquet": parquet_path, "csv": csv_path}
