# Notebooks

Pipeline notebooks that build the index, in execution order. Outputs are stripped; every statement is commented in English.

| Notebook | Purpose |
|---|---|
| `01_hazard_event_ingestion` | Clean the raw event table, validate dates and coordinates, remove personal data, build the monthly ENSO series |
| `02_spatial_integration` | Load official boundaries, cross-check each event's canton, build the event point layer |
| `03_historical_analysis` | Recurrence, impacts, ENSO-category frequency and trends per province and canton |
| `04_severity_index` | Severity index from five impact pillars |
| `05_recurrence_index` | Nino 1+2 vs ICEN/ONI, El Nino month definition and recurrence index |
| `06_persistence_index` | Runs of consecutive affected months |
| `07_enso_magnitude_index` | Strength of the El Nino episodes faced by each unit |
| `08_composite_index` | Composite chronicity index, Jenks classes and first sensitivity tests |
| `09_spatial_analysis` | Getis-Ord Gi*, Local Moran's I, kernel density, H3 hexagons |
| `11_sentinel1_flood_frequency_gee` | Sentinel-1 flood frequency in Google Earth Engine |
| `11_sar_flood_classification` | Classify and validate the SAR flood-frequency raster |
| `12_predictive_component` | Random Forest estimate of impacts per canton and episode |
| `13_sensitivity_validation` | Monte Carlo, Sobol, AHP, PCA, robust normalization and bootstrap |
| `15_enso_hazard_association` | Negative binomial association between El Nino and hazard events |

## Running them

Run the notebooks from this folder with the project environment (`pip install -r ../requirements.txt`). GitHub only displays notebooks; it does not execute them.

Most notebooks read the event-level table (`hazard_event*`) and the full boundary file, which are **not** included in this repository (restricted source data, see `config/data_sources.yaml`). Without access to them, use the notebooks as documentation of the method; the aggregated results they produce are already in `data/processed/`.

The Earth Engine notebook needs your own Google Earth Engine account: replace `YOUR_GEE_PROJECT_ID` with your project and provide the flood points file (derived from the restricted events).
