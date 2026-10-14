# Hazard Chronicity Index

**Índice de cronicidad de amenazas asociadas a El Niño en Ecuador (2015–2025)**

Identifica en qué cantones se repiten de forma crónica las inundaciones, deslizamientos, socavamientos y aluviones, qué tan graves son y cuánto se relacionan con los episodios de El Niño. Su propósito es apoyar la Acción Anticipatoria y la priorización territorial.

**Tablero interactivo:** https://jachicanoy.github.io/hazard-chronicity-index/

Autor: Jerson Achicanoy, Information Management and GIS Expert, 3iS LATAM.

## Qué incluye

| Carpeta | Contenido |
|---|---|
| `index.html` | Atlas interactivo, página principal de GitHub Pages |
| `src/`, `tests/`, `config/` | Código modular en Python, pruebas y configuración |
| `notebooks/` | Cuadernos del flujo completo, comentados en inglés y sin salidas (ver `notebooks/README.md`) |
| `data/processed/` | Resultados agregados por cantón y provincia (índices, hotspots, asociación ENSO) |
| `outputs/` | Mapas, figuras, tablas, tablero y reportes |
| `docs/` | Resumen ejecutivo e infografía del método |
| `scripts/` | Utilidades, incluida la descarga de datos pesados |

## Método en breve

Para cada cantón se combinan cuatro componentes en un índice de 0 a 1, clasificado en cinco niveles (de Muy Bajo a Muy Alto):

| Componente | Peso | Qué mide |
|---|---|---|
| Recurrencia | 35 % | Frecuencia de eventos, en especial en meses de El Niño |
| Severidad | 30 % | Peor impacto histórico en población, vivienda, agricultura, ganadería e infraestructura |
| Persistencia | 20 % | Duración de las rachas y número de meses del año afectados |
| Magnitud de El Niño | 15 % | Fuerza de los episodios enfrentados (índice ICEN del ENFEN) |

Los pesos se validaron con AHP, escenarios alternativos, simulación de Monte Carlo, índices de Sobol y PCA. El análisis se complementa con estadística espacial (Getis-Ord Gi*, Moran local, densidad de núcleo, hexágonos H3), detección de inundación con radar Sentinel-1 y un modelo predictivo de impactos.

## ¿Cuánto se debe a El Niño?

Un modelo binomial negativo, ajustado por estacionalidad y tendencia, compara los meses con y sin El Niño: la señal es fuerte en la Costa (×3.5, y ×5 en inundación), no detectable en la Sierra ni en el Oriente, e inversa en aluviones. El índice describe cronicidad histórica; **solo en la Costa** puede atribuirse con claridad a El Niño.

## Alcance y limitaciones

- No es un pronóstico de lluvia ni de impactos futuros.
- Solo hay 5 episodios de El Niño en el período; los resultados muestran asociación, no causa.
- Se basa en reportes SNGRE/ECU911, con diferencias de cobertura entre regiones y años.
- La región (Costa, Sierra, Oriente) se asigna por provincia; algunos cantones de transición pueden quedar mal clasificados.

## Datos

El repositorio guarda solo código, resultados agregados y documentación. Los datos de origen no se redistribuyen: los reportes SNGRE/ECU911 son de acceso restringido y los límites administrativos provienen de INEC/HDX. Los datos pesados (modelos, límites a nivel de parroquia) se declaran en `config/data_sources.yaml` y se obtienen con:

```bash
python scripts/fetch_data.py
```

## Instalación y pruebas

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest
```

Requiere Python 3.12.

## Licencia y cita

El código se distribuye bajo licencia MIT (`LICENSE`). La documentación, los resultados y el tablero, bajo CC BY 4.0 (`LICENSE-DOCS.md`). Para citar este trabajo, use `CITATION.cff`.

## Contacto

Jerson Achicanoy, jachicanoy@3is.org
