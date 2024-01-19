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
- `config_example.yaml`: contains an example of the configuration file needed to run the code. The file should be renamed to `config.yaml` and adapted to the specific use case.

## Data needed
For most of the operations, the following data is needed:
- A DEM (Digital Elevation Model) in GeoTIFF format
- Precipitation data in netCDF format. For now, only the [CombiPrecip](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/combiprecip.html) dataset is supported, but the code can be easily adapted to other datasets.
- To train the models, a dataset of surface water flood damages is needed. For now, claims data from the Swiss Mobiliar Insurance Company as GeoTIFF or from the GVZ (Building insurance Canton Zurich) as netCDF are supported. The code can be easily adapted to other datasets.


