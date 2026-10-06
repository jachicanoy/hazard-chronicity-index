"""Configuración estándar de logging para notebooks y módulos del proyecto.

Regla del proyecto (2026-08-05): usar logging, nunca ``print`` para trazar
el avance o los resultados de validación de un pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


def get_logger(
    name: str,
    config: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """Crea (o recupera) un logger configurado con salida a consola y a archivo.

    Args:
        name: Nombre del logger, normalmente ``__name__`` del módulo o un
            identificador legible del notebook (p. ej. ``"step1_ingestion"``).
        config: Configuración del proyecto cargada con
            :func:`src.utilidades.config_utils.load_config`. Si se provee,
            se usa ``config["logging"]["level"]`` y ``config["logging"]["log_filename"]``.
        log_dir: Directorio donde escribir el archivo de log. Si se omite y
            ``config`` tampoco trae ``paths.log_dir``, solo se registra en consola.

    Returns:
        Logger listo para usar, con handlers configurados una única vez
        (llamadas repetidas con el mismo ``name`` no duplican handlers).

    Example:
        >>> logger = get_logger("step1_ingestion", config)
        >>> logger.info("Filas cargadas: %d", 69718)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Ya configurado (p. ej. si la celda del notebook se re-ejecuta) —
        # evitamos duplicar handlers, que causaría líneas de log repetidas.
        return logger

    level_name = "INFO"
    log_filename = "efci_pipeline.log"
    if config is not None:
        level_name = config.get("logging", {}).get("level", level_name)
        log_filename = config.get("logging", {}).get("log_filename", log_filename)
        if log_dir is None:
            log_dir = config.get("paths", {}).get("log_dir")

    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)  # no falla si ya existe
        file_handler = logging.FileHandler(log_dir_path / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False  # evita que el root logger duplique la salida
    return logger
