"""Descarga los datos pesados declarados en config/data_sources.yaml.

Uso:
    python scripts/fetch_data.py            # lista las fuentes y su estado
    python scripts/fetch_data.py datos_publicos

Las fuentes con ``backend: manual`` solo muestran instrucciones. Las de
``gdrive`` requieren ``pip install gdown`` y que la carpeta sea accesible para
quien ejecuta el script (no se almacenan credenciales en el repositorio).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("fetch_data")
ROOT = Path(__file__).resolve().parents[1]


def load_sources() -> dict:
    """Lee las fuentes declaradas en config/data_sources.yaml."""
    with open(ROOT / "config" / "data_sources.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["sources"]


def fetch(name: str, spec: dict) -> None:
    """Obtiene una fuente según su backend.

    Args:
        name: Nombre de la fuente.
        spec: Especificación de la fuente.
    """
    backend = spec["backend"]
    target = ROOT / spec.get("target", "data/")
    if backend == "gdrive":
        folder = spec["gdrive_folder_id"]
        if folder == "PENDIENTE":
            logger.warning("%s: falta gdrive_folder_id en data_sources.yaml", name)
            return
        import shutil
        import tempfile

        import gdown  # import local: dependencia opcional

        # Se descarga a una carpeta temporal y se copia en el destino conservando
        # la estructura (models/, raw/, processed/), sin sobrescribir archivos vacios.
        with tempfile.TemporaryDirectory() as tmp:
            gdown.download_folder(id=folder, output=tmp, quiet=False, remaining_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tmp, target, dirs_exist_ok=True)
    elif backend == "remote":
        logger.info("%s: se lee directamente de %s", name, spec["url"])
    else:
        logger.info("%s: descarga manual. %s", name, spec["description"])


def main(argv: list[str]) -> int:
    """Punto de entrada de línea de comandos."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sources = load_sources()
    names = argv or list(sources)
    for n in names:
        if n not in sources:
            logger.error("Fuente desconocida: %s", n)
            return 1
        fetch(n, sources[n])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
