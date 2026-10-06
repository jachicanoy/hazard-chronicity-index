"""Aviso metodológico del dashboard.

Inyecta, de forma idempotente, un aviso colapsable con el alcance, los
supuestos y las limitaciones del análisis en los HTML del dashboard y del
atlas. Las cifras de la tabla regional se leen de los resultados de la
etapa de asociación ENSO–amenaza (no se escriben a mano), para que el aviso nunca quede
desactualizado respecto del análisis.

El texto del aviso es la versión pública del "Texto del aviso para el
dashboard" de ``docs/executive/15_enso_hazard_association.md``.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MARK_START = "<!--AVISO_METODOLOGICO_START-->"
MARK_END = "<!--AVISO_METODOLOGICO_END-->"

TEXTO_PRINCIPAL = (
    "Este índice mide la <b>cronicidad histórica (2015–2025)</b> de eventos de "
    "inundación, deslizamiento, socavamiento y aluvión en los cantones de "
    "Ecuador, y su coincidencia con meses de El Niño. <b>No atribuye cada evento "
    "a El Niño:</b> una parte importante responde a la estacionalidad y a "
    "factores locales. El análisis estadístico muestra que El Niño se asocia "
    "claramente con más eventos en la Costa (en especial inundaciones), pero "
    "esa asociación es débil o no detectable en la Sierra y el Oriente, y los "
    "aluviones no aumentan durante El Niño."
)
TEXTO_ALCANCE = (
    "<b>El índice no es un pronóstico de lluvia ni de impactos futuros:</b> "
    "supone que los patrones históricos pueden repetirse, algo menos sólido "
    "donde la señal de El Niño es débil. Se basa en reportes SNGRE/ECU911, "
    "sujetos a variaciones de cobertura entre regiones y años; solo hay 5 "
    "episodios de El Niño en el período, y los resultados describen "
    "asociación, no causa. Úsese como apoyo a la priorización junto con "
    "información climática vigente (p. ej. INAMHI) y análisis local, no como "
    "sustituto."
)

CSS = """
<style id="aviso-metodologico-css">
#aviso-metodologico{font-family:'Montserrat',Arial,sans-serif;background:#FFF6E5;
 border-bottom:3px solid #F4B942;color:#2e2e2e;padding:10px 24px;font-size:13px;line-height:1.5}
#aviso-metodologico summary{cursor:pointer;font-weight:700;color:#063B6D;font-size:13px}
#aviso-metodologico .am-body{max-width:1100px;margin-top:8px}
#aviso-metodologico table{border-collapse:collapse;margin:8px 0;font-size:12px}
#aviso-metodologico th{background:#063B6D;color:#fff;text-align:left;padding:4px 10px}
#aviso-metodologico td{padding:4px 10px;border-bottom:1px solid #e0d6c0}
#aviso-metodologico .am-nota{font-size:11px;color:#667580}
</style>
"""


def _tabla_regional(irr: pd.DataFrame) -> str:
    """Construye la tabla HTML de señal por región/amenaza desde los resultados.

    Args:
        irr: Tabla ``enso_asociacion_irr`` (modelo primario con ``p_BH``).

    Returns:
        Fragmento HTML ``<table>`` con IRR, IC 95 % y significancia.
    """
    prim = irr[irr["modelo"] == "M1_primario_ICEN_binario"].set_index("estrato")
    filas = [
        ("Costa", "Región Costa"), ("Sierra", "Región Sierra"), ("Oriente", "Región Oriente"),
        ("Inundación", "Amenaza flood"), ("Deslizamiento", "Amenaza landslide"),
        ("Socavamiento", "Amenaza scour"), ("Aluvión", "Amenaza debris_flow"),
    ]
    cuerpo = []
    for etiqueta, clave in filas:
        r = prim.loc[clave]
        if not bool(r["significativo_BH"]):
            lectura = "no detectable"
        elif r["IRR"] < 1:
            lectura = "menos eventos con El Niño"
        else:
            lectura = "más eventos con El Niño"
        cuerpo.append(
            f"<tr><td>{html.escape(etiqueta)}</td><td>×{r['IRR']:.2f}</td>"
            f"<td>{r['IC95_inf']:.2f}–{r['IC95_sup']:.2f}</td><td>{lectura}</td></tr>"
        )
    return (
        "<table><tr><th>Estrato</th><th>Factor de eventos/mes con El Niño</th>"
        "<th>IC 95 %</th><th>Lectura</th></tr>" + "".join(cuerpo) + "</table>"
    )


def build_banner(irr: pd.DataFrame) -> str:
    """Genera el bloque HTML completo (CSS + aviso) listo para inyectar.

    Args:
        irr: Tabla de resultados de la etapa de asociación ENSO–amenaza.

    Returns:
        Cadena HTML delimitada por marcadores para poder reemplazarla.
    """
    cuerpo = (
        '<details id="aviso-metodologico"><summary>Alcance, supuestos y limitaciones '
        "de este análisis (haga clic para leer)</summary><div class=\"am-body\">"
        f"<p>{TEXTO_PRINCIPAL}</p>{_tabla_regional(irr)}"
        '<p class="am-nota">Factor ajustado por estacionalidad y tendencia '
        "(regresión binomial negativa, 2015–2025); 1 = sin efecto.</p>"
        f"<p>{TEXTO_ALCANCE}</p></div></details>"
    )
    return f"{MARK_START}{CSS}{cuerpo}{MARK_END}"


def inject_banner(html_path: str | Path, irr: pd.DataFrame) -> bool:
    """Inserta (o reemplaza) el aviso metodológico en un archivo HTML.

    Es idempotente: si ya existe un aviso entre marcadores, lo reemplaza, de
    modo que reejecutar tras actualizar la etapa de asociación ENSO–amenaza no duplica contenido. Se
    inserta justo después de ``</header>`` o, si no existe, tras ``<body>``.
    Se conserva la marca de orden de bytes (BOM) del archivo original.

    Args:
        html_path: Ruta del HTML.
        irr: Tabla de resultados de la etapa de asociación ENSO–amenaza.

    Returns:
        ``True`` si el archivo fue modificado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si no se encuentra dónde insertar el aviso.
    """
    ruta = Path(html_path)
    if not ruta.exists():
        raise FileNotFoundError(ruta)
    crudo = ruta.read_bytes()
    tiene_bom = crudo.startswith(b"\xef\xbb\xbf")
    texto = crudo.decode("utf-8-sig")
    banner = build_banner(irr)

    patron = re.compile(re.escape(MARK_START) + r".*?" + re.escape(MARK_END), re.S)
    if patron.search(texto):
        nuevo = patron.sub(lambda _m: banner, texto, count=1)
    else:
        m = re.search(r"</header>", texto, re.I) or re.search(r"<body[^>]*>", texto, re.I)
        if m is None:
            raise ValueError(f"No se encontró </header> ni <body> en {ruta}")
        nuevo = texto[: m.end()] + "\n" + banner + "\n" + texto[m.end():]

    if nuevo == texto:
        return False
    salida = nuevo.encode("utf-8")
    ruta.write_bytes((b"\xef\xbb\xbf" if tiene_bom else b"") + salida)
    logger.info("Aviso metodológico inyectado en %s", ruta)
    return True
