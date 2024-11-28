# SWAFI - Surface Water Flood Impact

## Install the package
Install the released version of the package using the following command:
```
pip install swafi
```

## Installing dependencies
Install the requirements using the following command:
```
pip install -r requirements.txt
```
Additional requirements needed for specific operations can be installed using the following commands:
```
pip install -r requirements-optional.txt
```

## Structure of the repository
The repository is structured as follows:
- `files`: contains a file (`cids.tif`) with the IDs of the precipitation dataset cells.
- `swafi`: contains the code of the package with the most important functions.
- `scripts`: contains scripts to run different operations. They can be used as a starting point to run the code.
  - `data_analyses`: contains scripts to perform data analyses.
  - `data_preparation`: contains scripts to prepare the data for the different operations.
  - `impact_functions`: contains scripts to train and evaluate the different impact functions.
  - `link_claims_events`: contains scripts to link claims and events.
- `config_example.yaml`: contains an example of the configuration file needed to run the code. 
  The file should be renamed to `config.yaml` and adapted to the specific use case.

## Data needed
For most of the operations, the following data is needed:
- A DEM (Digital Elevation Model) in GeoTIFF format
- Precipitation data in netCDF format. For now, only the 
  [CombiPrecip](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/combiprecip.html) 
  dataset is supported, but the code can be easily adapted to other datasets.
- To train the models, a dataset of surface water flood damages is needed. 
  For now, claims data from the Swiss Mobiliar Insurance Company as GeoTIFF or from the 
  GVZ (Building insurance Canton Zurich) as netCDF are supported. 
  The code can be easily adapted to other datasets.

## Main components

### Events

The events are precipitation events per cell extracted from the gridded precipitation dataset (CombiPrecip).
These are managed by the `Events` class in `swafi/events.py` and are handled internally as a Pandas dataframe.
Different characteristics are calculated for each event:
- `e_start`: start date of the event
- `e_end`: end date of the event
- `duration`: duration of the event in hours
- `duration_q`: quantile equivalent for the duration
- `p_sum`: total precipitation of the event in mm
- `p_sum_q`: quantile equivalent for the total precipitation
- `i_max`: maximum precipitation intensity of the event in mm/h
- `i_max_q`: quantile equivalent for the maximum precipitation intensity
- `i_mean`: mean precipitation intensity of the event in mm/h
- `i_mean_q`: quantile equivalent for the mean precipitation intensity
- `i_sd`: standard deviation of the precipitation intensity of the event in mm/h
- `i_sd_q`: quantile equivalent for the standard deviation of the precipitation intensity
- `api`: antecedent precipitation index for the event
- `api_q`: quantile equivalent for the antecedent precipitation index

### Damages

The damages correspond to insurance claims per cell (pixel of the precipitation dataset).
The damages are managed by the `Damages` class in `swafi/damages.py` and are also handled internally as a Pandas dataframe.

There are two classes of damages:

#### DamagesMobiliar
`DamagesMobiliar` from the file `swafi/damages_mobiliar.py`: handles the claims from the Swiss Mobiliar Insurance Company as GeoTIFF.
The dataset from the Mobiliar contains the following categories of **exposure** (contracts):

| Name in swafi       | Client  | Ext/Int  | Object    | Original file names         |
|---------------------|---------|----------|-----------|-----------------------------|
| sme_ext_cont        | SME     | external | content   | Vertraege_KMU_ES_FH_YYYY    |
| sme_ext_struc       | SME     | external | structure | Vertraege_KMU_ES_GB_YYYY    |
| sme_int_cont        | SME     | internal | content   | Vertraege_KMU_W_FH_YYYY     |
| sme_int_struc       | SME     | internal | structure | Vertraege_KMU_W_GB_YYYY     |
| priv_ext_cont       | Private | external | content   | Vertraege_Privat_ES_FH_YYYY |
| priv_ext_struc      | Private | external | structure | Vertraege_Privat_ES_GB_YYYY |
| priv_int_cont       | Private | internal | content   | Vertraege_Privat_W_FH_YYYY  |
| priv_int_struc      | Private | internal | structure | Vertraege_Privat_W_GB_YYYY  |

The dataset from the Mobiliar contains the following categories of **claims**:

| Name in swafi       | Client  | Ext/Int  | Object    | Flood type | Original file names               |
|---------------------|---------|----------|-----------|------------|-----------------------------------|
| sme_ext_cont_pluv   | SME     | external | content   | pluvial    | Ueberschwemmung_pluvial_KMU_FH    |
| sme_ext_cont_fluv   | SME     | external | content   | fluvial    | Ueberschwemmung_fluvial_KMU_FH    |
| sme_ext_struc_pluv  | SME     | external | structure | pluvial    | Ueberschwemmung_pluvial_KMU_GB    |
| sme_ext_struc_fluv  | SME     | external | structure | fluvial    | Ueberschwemmung_fluvial_KMU_GB    |
| sme_int_cont        | SME     | internal | content   |            | Wasser_KMU_FH                     |
| sme_int_struc       | SME     | internal | structure |            | Wasser_KMU_GB                     |
| priv_ext_cont_pluv  | Private | external | content   | pluvial    | Ueberschwemmung_pluvial_Privat_FH |
| priv_ext_cont_fluv  | Private | external | content   | fluvial    | Ueberschwemmung_fluvial_Privat_FH |
| priv_ext_struc_pluv | Private | external | structure | pluvial    | Ueberschwemmung_pluvial_Privat_GB |
| priv_ext_struc_fluv | Private | external | structure | fluvial    | Ueberschwemmung_fluvial_Privat_GB |
| priv_int_cont       | Private | internal | content   |            | Wasser_Privat_FH                  |
| priv_int_struc      | Private | internal | structure |            | Wasser_Privat_GB                  |

#### DamagesGVZ
`DamagesGVZ` from the file `swafi/damages_gvz.py`: handles the claims from the GVZ (Building insurance Canton Zurich) as netCDF.
The dataset from the GVZ contains a single category of **exposure** (contracts): `all_buildings`, and the following categories of **claims**:

| Name in swafi       | Original tag  |
|---------------------|---------------|
| most_likely_pluvial | A             |
| likely_pluvial      | A, B          |
| fluvial_or_pluvial  | A, B, C, D, E |
| likely_fluvial      | D, E          |
| most_likely_fluvial | E             |

These classes are subclasses of the `Damages` class and implement the data loading according to the corresponding file format as well as their specific classification.

### Impact

These are the impact functions aiming to predict the damages from the precipitation events.
The impact functions are managed by the subclasses of the `Impact` class in `swafi/impact.py`.
The subclasses implement the training and evaluation of the impact functions as well as the prediction of the damages.
The following impact functions are implemented:
- `ImpactThresholds` from the file `swafi/impact_thr.py`: predicts a damage if the precipitation exceeds certain thresholds.
  This is the approach introduced by [Bernet et al. (2019)](https://dx.doi.org/10.1088/1748-9326/ab127c).
- `ImpactLogisticRegression` from the file `swafi/impact_lr.py`: predicts the damages using a regression model.
- `ImpactRandomForest` from the file `swafi/impact_rf.py`: predicts the damages using a random forest model.
- `ImpactDeepLearning` from the file `swafi/impact_dl.py`: predicts the damages using a deep learning model.
  The deep-learning model itself (`DeepImpact`) is implemented in the file `model_dl.py` using the Keras and Tensorflow libraries. 

## Workflow

The workflow is structured in the steps described below.
Before running the code, the configuration file `config.yaml` should be created to define the paths to the data and the parameters of the different operations.
An example of the configuration file is provided in `config_example.yaml`.
Pickle files are used to save the results of the different operations to avoid having to re-run some time-consuming operations.

### 1. Extracting events

The first step is to extract the events from the precipitation dataset.
The events are extracted for the whole domain using the `extract_precipitation_events.py` 
script in `scripts/data_preparation` and saved as a parquet files.

### 2. Linking events and damages

The second step is to link the events and the damages.
The events and damages are linked using the `compute_claims_events_link.py` script in `scripts/link_claims_events`.

This script will:

1. Load the original contract and claims data (in their respective format) and save them as pickle files.
   The operation is performed only if the pickle files do not exist yet.

2. Select the claims and contracts for the selected categories. The selections by default are:
   - Mobiliar data: 
     - Exposure category: `external`
     - Claim categories: `['external', 'pluvial']`
   - GVZ data: 
     - Exposure category: `all_buildings`
     - Claim categories: `['likely_pluvial']`
   
   The selection can be changed by modifying the `EXPOSURE_CATEGORIES` and `CLAIM_CATEGORIES` variable in the script.
   The selected claims and contracts are saved as pickle files.

3. Select the events where we have insurance data, i.e. removing all cells where we do not have insurance data. 
   The selection is performed using the `load_events_and_select_those_with_contracts` function in `scipts/data_preparation`.
   The selected events are saved as pickle files.

4. Link the damages to the events using the `link_with_events` function.
   The function will try to match the claim date with the event date by picking the most likely event.
   The procedure is the same as in [Bernet et al. (2019)](https://dx.doi.org/10.1088/1748-9326/ab127c).
   The event attributes used to link the events and damages can be set in the `CRITERIA` variable in the script.
   The default criteria are: `['prior', 'i_mean', 'i_max', 'p_sum', 'r_ts_win', 'r_ts_evt']` 
   (the `prior` criteria gives more weight to events happening before the damage occurrence).
   The temporal windows used to look for events can be set in the `WINDOW_DAYS` variable in the script.
   The default windows are: `[5, 3, 1]`.
   The linked damages are saved as pickle files with the name `damages_{DATASET}_linked_{LABEL}.pickle`.

5. Assign the target value to the damages, depending if the focus is on the `occurrence` or the `damage_ratio`.

6. From the selection of events on step 3, assign the target value to the events based on the corresponding damages.
   All events not linked to damages are assigned a target value of 0.
   The selected events are saved as pickle files with the name `events_{DATASET}_with_target_values_{LABEL_RESULTING_FILE}`.

The file resulting from this step is a pickle file with the events and the assigned target values.
This file is needed to train or assess the impact functions.

### 3. Preparing static attributes

Some static attributes can be computed for each cell of the precipitation dataset.
Some of these rely on DEM data, which should be provided in GeoTIFF format.
The attributes can be computed on DEMs of different resolutions (e.g., 10m, 25m, 50m, 100m, 250m) and later aggregated to the resolution of the precipitation dataset.
Therefore, different DEMS can be provided in the configuration file.
Each attribute name will then contain the DEM resolution (e.g., `dem_010m_flowacc`).
The following attributes can be computed:

- Flow accumulation (using the PyShed library as in `compute_flow_accumulation_pysheds.py` or the RichDEM library as in `compute_flow_accumulation_richdem.py`):
  - `dem_{RES}_flowacc`: flow accumulation from the DEM of resolution `{RES}`.
  - `dem_{RES}_flowacc_nolakes`: flow accumulation from the DEM of resolution `{RES}` with lakes removed.
  - `dem_{RES}_flowacc_norivers`: flow accumulation from the DEM of resolution `{RES}` with rivers and lakes removed.
- Terrain (using `compute_terrain_attributes.py`):
  - `dem_{RES}_aspect`: aspect from the DEM of resolution `{RES}`.
  - `dem_{RES}_curv_plan`: plan curvature from the DEM of resolution `{RES}`.
  - `dem_{RES}_curv_prof`: profile curvature from the DEM of resolution `{RES}`.
  - `dem_{RES}_curv_tot`: total curvature from the DEM of resolution `{RES}`.
  - `dem_{RES}_slope`: slope from the DEM of resolution `{RES}`.
- Topographic wetness index (using `compute_topographic_wetness_index.py`):
  - `dem_{RES}_twi`: topographic wetness index from the DEM of resolution `{RES}`.

These scripts will generate GeoTIFF files with the computed attributes.
These attributes can then be extracted for each GeoTIFF file using the `extract_static_data_to_csv.py` script.
The aggregation to the resolution of the precipitation dataset is performed using different statistics: `min`, `max`, `mean`, `std`, `median`.
The attributes are then saved as csv files adding the statistics to the attribute name (e.g., `dem_{RES}_flowacc_{STAT}`).

Other attributes can be used as well by extracting them from other GeoTIFF files.
Examples of such attributes are: 
- `swf_map`: categories from the surface water flood map.
- `land_cover`: coverage from each land cover class from the land cover map.
- `runoff_coeff`: runoff coefficient computed from the land cover map.

### 4. Training and evaluating the impact functions

The different impact functions are trained and assessed using relatively similar scripts.
These scripts are located in `scripts/impact_functions`.

For example, the code for training the logistic regression model is:

```python
from swafi.events import load_events_from_pickle
from swafi.impact_lr import ImpactLogisticRegression

# Load the events with the target values
events_filename = f'events_mobiliar_with_target_values_pluvial_occurrence.pickle'
events = load_events_from_pickle(filename=events_filename)

# Create the impact function
lr = ImpactLogisticRegression(events)

# Load the desired attributes
lr.load_features(['event', 'terrain', 'swf_map', 'runoff_coeff'])

# Split the data into training, validation, and test sets
lr.split_sample()
# Normalize the features
lr.normalize_features()
# Compute the class weights (claim/no claim)
lr.compute_balanced_class_weights()
# Decrease the weight of the events with claims by a certain factor (otherwise it will be too high)
lr.compute_corrected_class_weights(weight_denominator=27)
# Train the model
lr.fit()
# Evaluate the model on all splits
lr.assess_model_on_all_periods()
```

The functions `split_sample`, `normalize_features`, `compute_balanced_class_weights`, 
`compute_corrected_class_weights`, and `assess_model_on_all_periods` are common to all impact functions.
The function `fit` is specific to each impact function.

#### Thresholds model

The thresholds model is trained and evaluated using the `apply_thresholds_v2019.py` script.
It requires the following data:
- The events with the target values computed in step 2.

#### Logistic regression model

The logistic regression model is trained and evaluated using the `train_lr_occurrence.py` script.
It requires the following data:
- The events with the target values computed in step 2.
- The static attributes computed in step 3 (optional).

#### Random forest model

The random forest model is trained and evaluated using the `train_rf_occurrence.py` script.
An optimisation of the hyperparameters is performed using the `train_rf_occurrence_hyperparameters.py` script.
They requires the following data:
- The events with the target values computed in step 2.
- The static attributes computed in step 3 (optional).

#### Deep learning model

The deep learning model is trained and evaluated using the `train_dl_occurrence.py` script.
it requires the following data:
- The events with the target values computed in step 2.
- The static attributes computed in step 3 (optional).
- The original precipitation data in netCDF format.

## Training the deep learning model

The deep learning model is implemented in the `DeepImpact` class in the `model_dl.py` file.
The model is built using the Keras and Tensorflow libraries.
The model can be trained using the `train_dl_occurrence.py` script.
The different input data (precipitation, DEM, static features) can be turned on or off.
All the hyperparameters of the model can be set as options of the script.
The model can be trained using the following command:

```bash
train_cnn_occurrence.py [-h] [--run-id RUN_ID] [--optimize-with-optuna]
                       [--target-type TARGET_TYPE]
                       [--factor-neg-reduction FACTOR_NEG_REDUCTION]
                       [--weight-denominator WEIGHT_DENOMINATOR]
                       [--random-state RANDOM_STATE]
                       [--do-not-use-precip] [--do-not-use-dem]
                       [--do-not-use-simple-features]
                       [--simple-feature-classes SIMPLE_FEATURE_CLASSES [SIMPLE_FEATURE_CLASSES ...]]
                       [--simple-features SIMPLE_FEATURES [SIMPLE_FEATURES ...]]
                       [--precip-window-size PRECIP_WINDOW_SIZE]
                       [--precip-resolution PRECIP_RESOLUTION]
                       [--precip-time-step PRECIP_TIME_STEP]
                       [--precip-days-before PRECIP_DAYS_BEFORE]
                       [--precip-days-after PRECIP_DAYS_AFTER]
                       [--transform-static TRANSFORM_STATIC]
                       [--transform-2d TRANSFORM_2D]
                       [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                       [--learning-rate LEARNING_RATE]
                       [--dropout-rate DROPOUT_RATE]
                       [--no-spatial-dropout] [--no-batchnorm]
                       [--nb-filters NB_FILTERS]
                       [--nb-conv-blocks NB_CONV_BLOCKS]
                       [--nb-dense-layers NB_DENSE_LAYERS]
                       [--nb-dense-units NB_DENSE_UNITS]
                       [--no-dense-units-decreasing]
                       [--inner-activation INNER_ACTIVATION]
```

Options
* `-h`, `--help`: show this help message and exit
* `--run-id RUN_ID`: The run ID
* `--optimize-with-optuna`: Optimize the hyperparameters with Optuna
* `--target-type TARGET_TYPE`: The target type. Options are: occurrence, damage_ratio
* `--factor-neg-reduction FACTOR_NEG_REDUCTION`: The factor to reduce the number of negatives only for training
* `--weight-denominator WEIGHT_DENOMINATOR`: The weight denominator to reduce the negative class weights
* `--random-state RANDOM_STATE`: The random state to use for the random number generator
* `--do-not-use-precip`: Do not use precipitation data
* `--do-not-use-dem`: Do not use DEM data
* `--do-not-use-simple-features`: Do not use simple features (event properties and static attributes)
* `--simple-feature-classes SIMPLE_FEATURE_CLASSES [SIMPLE_FEATURE_CLASSES ...]`: The list of simple feature classes to use (e.g. event terrain)
* `--simple-features SIMPLE_FEATURES [SIMPLE_FEATURES ...]`: The list of specific simple features to use (e.g. event:i_max_q). If not specified, the default class features will be used. If specified, the default class features will be overridden for this class only (e.g. event).
* `--precip-window-size PRECIP_WINDOW_SIZE`: The precipitation window size [km]
* `--precip-resolution PRECIP_RESOLUTION`: The precipitation resolution [km]
* `--precip-time-step PRECIP_TIME_STEP`: The precipitation time step [h]
* `--precip-days-before PRECIP_DAYS_BEFORE`: The number of days before the event to use for the precipitation
* `--precip-days-after PRECIP_DAYS_AFTER`: The number of days after the event to use for the precipitation
* `--transform-static TRANSFORM_STATIC`: The transformation to apply to the static data
* `--transform-2d TRANSFORM_2D`: The transformation to apply to the 2D data
* `--batch-size BATCH_SIZE`: The batch size
* `--epochs EPOCHS`: The number of epochs
* `--learning-rate LEARNING_RATE`: The learning rate
* `--dropout-rate DROPOUT_RATE`: The dropout rate
* `--no-spatial-dropout`: Do not use spatial dropout
* `--no-batchnorm`: Do not use batch normalization
* `--nb-filters NB_FILTERS`: The number of filters
* `--nb-conv-blocks NB_CONV_BLOCKS`: The number of convolutional blocks
* `--nb-dense-layers NB_DENSE_LAYERS`: The number of dense layers
* `--nb-dense-units NB_DENSE_UNITS`: The number of dense units
* `--no-dense-units-decreasing`: The number of dense units should not decrease
* `--inner-activation INNER_ACTIVATION`: The inner activation function

When using Optuna, all but the following options are opimized: `run-id`,
`target-type`, `random-state`, `factor-neg-reduction`,
`do-not-use-precip`, `do-not-use-dem`, `do-not-use-simple-features`, 
`simple-feature-classes`, `simple-features`.

## Applying the impact functions to other precipitation datasets

