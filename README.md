# SWAFI - Surface Water Flood Impact

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
- `e_tot`: duration of the event in hours
- `e_tot_q`: quantile equivalent for the duration
- `p_sum`: total precipitation of the event in mm
- `p_sum_q`: quantile equivalent for the total precipitation
- `i_max`: maximum precipitation intensity of the event in mm/h
- `i_max_q`: quantile equivalent for the maximum precipitation intensity
- `i_mean`: mean precipitation intensity of the event in mm/h
- `i_mean_q`: quantile equivalent for the mean precipitation intensity
- `i_sd`: standard deviation of the precipitation intensity of the event in mm/h
- `i_sd_q`: quantile equivalent for the standard deviation of the precipitation intensity
- `apireg`: antecedent precipitation index for the event
- `apireg_q`: quantile equivalent for the antecedent precipitation index

### Damages

The damages correspond to insurance claims per cell (pixel of the precipitation dataset).
The damages are managed by the `Damages` class in `swafi/damages.py` and are also handled internally as a Pandas dataframe.

There are two classes of damages:
- `DamagesMobiliar` from the file `swafi/damages_mobiliar.py`: handles the claims from the Swiss Mobiliar Insurance Company as GeoTIFF.
  The dataset from the Mobiliar contains the following categories of claims:

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

- `DamagesGVZ` from the file `swafi/damages_gvz.py`: handles the claims from the GVZ (Building insurance Canton Zurich) as netCDF.
  The dataset from the GVZ contains the following categories:

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
- `ImpactThresholds` from the file `swafi/impact_thr.py`: predicts a damage if the precipitation exceeds a certain threshold.




The events are extracted for the whole domain using the `extract_precipitation_events.py` script in `scipts/data_preparation` and saved as a parquet files.
Then, as we are interested in the events where we have insurance data, we filter the events using the `load_events_and_select_those_with_contracts` function in `scipts/data_preparation`.
