# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SWAFI (Surface Water Flood Impact) is a Python package for predicting surface water flood damages from precipitation events in Switzerland. It links precipitation events from CombiPrecip to insurance claims (Swiss Mobiliar and GVZ — Canton Zurich Building Insurance) and trains impact models ranging from simple thresholds to deep learning.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
pip install -e .
cp config_example.yaml config.yaml
# Edit config.yaml with paths to data directories
```

The `config.yaml` file must be present in the working directory (or one or two levels up). It specifies all data paths (`OUTPUT_DIR`, `PICKLES_DIR`, `TMP_DIR`, `EVENTS_PATH`, `DIR_PRECIP`, exposure/claims dirs, DEM paths, static attribute CSVs, etc.).

## Running Tests

```bash
pytest tests/
pytest tests/test_events_pickle_loading.py -v
```

## Code Formatting

Black (line-length 88) and isort (black profile) are configured in `pyproject.toml`.

## Architecture

The package lives in `swafi/`. The workflow proceeds through these stages:

1. **Precipitation events** — extracted from CombiPrecip (netCDF) into a parquet file
2. **Linking claims to events** — spatiotemporal matching of insurance claims to precipitation events
3. **Static attributes** — computed from DEMs (terrain, flow accumulation, TWI, land cover, runoff coefficients)
4. **Impact model training** — using the linked events+claims dataset
5. **Inference** — applying trained models

### Core Classes

**`Events`** (`swafi/events.py`)
Wraps a Pandas DataFrame of precipitation events. Columns include: `cid`, `x`, `y`, `e_start`, `e_end`, `duration`, `p_sum`, `i_max`, `i_mean`, `i_sd`, `api` — plus quantile-normalized variants (e.g. `i_max_q`). Uses pickle-based caching with fallback compatibility.

**`Damages`** (`swafi/damages.py`) + subclasses
Base class for insurance exposure and claims data.
- `DamagesMobiliar` (`damages_mobiliar.py`): reads GeoTIFF format from Swiss Mobiliar
- `DamagesGVZ` (`damages_gvz.py`): reads netCDF format from GVZ

**`Impact`** (`swafi/impact.py`) + subclasses
Base class for all impact prediction models. Subclasses:
- `ImpactThresholds` (`impact_thr.py`): simple precipitation threshold approach
- `ImpactLogisticRegression` (`impact_lr.py`): scikit-learn LR
- `ImpactRandomForest` (`impact_rf.py`): RF with Optuna hyperparameter optimization
- `ImpactDl` (`impact_dl.py`): base for deep learning models
  - `ImpactCnn` (`impact_cnn.py`): CNN with spatial/temporal input
  - `ImpactTx` (`impact_tx.py`): Transformer model

**`Precip`** (`swafi/precip.py`) + subclasses
Handles precipitation data reading.
- `PrecipCombiPrecip` (`precip_combiprecip.py`): MeteoSwiss CombiPrecip product
- `PrecipArchive` / `PrecipForecast` / `PrecipIcon`: other precipitation sources

**`Config`** (`swafi/config.py`)
Singleton-style YAML config loader. Searches for `config.yaml` in the current or parent directories (up to two levels). Access via `config.get('KEY')`.

### Feature Selection

Features are specified in `'class:feature'` format, e.g. `'event:i_max_q'`, `'terrain:slope'`. The `Impact.select_features()` method parses these and sets per-class feature lists. Feature classes include: `event`, `terrain`, `flowacc`, `twi`, `runoff_coeff`, `land_cover`, `swf`.

### Deep Learning Data Pipeline

`ImpactCnnDataGenerator` and `ImpactTxDataGenerator` (`impact_cnn_data_generator.py`, `impact_tx_data_generator.py`) produce batches for CNN and Transformer models respectively. CNNs expect 3D arrays (spatial window × time steps). Options are managed via `ImpactCnnOptions` / `ImpactTxOptions`.

### Scripts

`scripts/` contains standalone analysis and training scripts organized by stage:
- `data_preparation/`: event extraction, precipitation time series, static attribute computation
- `link_claims_events/`: computing and analyzing the claims-events linkage
- `impact_functions/`: training scripts (`train_*.py`), inference scripts (`use_*.py`), and assessment scripts (`assess_*.py`)
- `plotting/`: visualization scripts
- `data_analyses/`: exploratory analyses