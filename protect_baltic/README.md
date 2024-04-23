# Sufficiency of Measures model

This project continues on the work done on the SOM model in the [HELCOM BLUES](https://github.com/helcomsecretariat/SOM/tree/main/helcom_blues) project as part of the [HELCOM PROTECT BALTIC](https://protectbaltic.eu/) project. 

Contains two programs:
- HELCOM API
    - API consumer to retrieve and process spatial data in raster or vector form
- SOM
    - Processes, organises and calculates data for the Sufficiency of Measures (SOM) analysis done at HELCOM

## Installation

Developed in Python version 3.12

Create an environment:

```
cd /path/to/protect_baltic
python -m venv .
source bin/activate
```

Editable version:

```
python -m pip install -e .  
```

Run only version:

```
python -m pip install .
```

## helcom_api

Contains the HELCOM API consumer. Figure 1 details the workflow of processing of fetched spatial data.

![image](docs/SOM-app-helcom-api.png)
Figure 1. Workflow of processing spatial data.

Currently requires the directory ```EEZ``` which contains topology corrected spatial shape files for exclusive economic zones and administrative boundaries. These are not available through the HELCOM API, but are given upon request.

```configuration.toml``` contains meta data of spatial data and other information needed for processing.

```gis_tools.py``` contains io and processing functions of the workflow.


## som

The scientific foundation of the SOM model framework can be found [here](https://helcom.fi/baltic-sea-action-plan/som/).

### Development - Part 1

Two of three input data files are used to initialize and populate the object model by programatically describing the problem domain, i.e. interactions between societal, economic and environment in its geographcical context in the Baltic Sea region. 

When constructing the model three principles are followed whenever possible:

 1. modularity
 2. objects with single responsibility
 3. clear interfaces
 
 Although the current version is a proof of concept of the object model, care has been taken to ensure it can be extended in future versions, e.g., expected values extracted from survey data and description of the problem domain are single values with a range (min and max), not distributions, but implementation of distributions is already planned in future versions.

Input data consists of three files:

- generalInput.xlsx
- measureEffInput.xslx
- pressStateInput.

```generalInput.xlsx``` contains descriptions of the problem domain:

- in ```sheet:ActMeas``` are governance issues, all rows are independent
    1. ID - ID of the case
    2. MT_ID - Measure type ID
    3. In_Activities - Relevant Activities, 0 means all relevant activities for Measure type in ```sheet:MT_to_A_to_P```
    4. In_Pressure - Relevant Pressures, 0 means all relevant pressures for Measure type in ```sheet:MT_to_A_to_P```
    5. In_State_components - Relevant States, 0 means all relevant states for Measure type in ```sheet:MT_to_A_to_P```
    6. Multiplier_OL - Measure type multiplier
    7. B_ID - Basin IDs
    8. C_ID - Country IDs
- in ```sheet:ActPres``` are environmental issues, lists the relevant basins for each Activity-Pressure pairs
    1. Activity - Activity ID
    2. Pressure - Pressure ID
    3. Basin - Basin ID
    4. Ml# - MostLikely (ActivityPressure.AP_expected)
    5. Min# - Minimum end of range
    6. Max# - Maximum end of range
- in ```sheet:MT_to_A_to_P``` are linkages between measure types, activities, pressures and states

```measureEffInput.xslx``` contains survey data on the effects of measures on activity-pressure pairs as surved by expert panels.

```pressStateInput.xlsx``` contains survey data on pressure reduction targets.

Model file structure (under ```src:som```):

- ```__main__```: calls facade functions that represents workflow steps. 
- ```configuration.toml```: holds paths to input data files and other meta data about input files.
- ```som_app.py```: contains facade functions that wrap underlying functions on thematic collections representing workflow steps.
- ```som_classes.py```: contains object definitions of core and secondary object layers.
- ```som_tools.py```: contains data reading and processing functions. 

## SOM object model

The model framework encapsulates the problem domain and survey data as a layered structure (Figure 2). 

![image](docs/SOM-app-class-diagram.png)
Figure 2. SOM class diagram.

The layers consist of a set of objects that describe basic elements of the problem domain and their interactions. Each object contains appropriate data either from the problem domain description, calculated from survey data, or calculated from all of the previous. 

The object model describes the core functionality of the SOM analysis, object oriented data structure as a result of expert panel survey data coupled with the problem domain description. It concentrates on model Measure, Activity-Pressure, Activity and Pressure interactions and stores approriate data. State, Pressure, and State-Pressure are included in object model but not yet implemented. 

In the initialization of the object model the following steps are taken:

1. Process survey data from measure effect input and read general input
2. Build core object model and initialize core object instances
3. Build second object layer model and initialize second object layer instances
4. Post-process core objects organised in secondary object layer by storing data into them from general input

The data flow from input data to data structures containing object model is shown in Figure 3. The diagram shows processes and data structures that are created upon initalization.

![image](docs/SOM-app-data-flow-diagram.png)
Figure 3. Data flow diagram. 

After the initialization, altogether 41 object models are instantiated. They represents 41 country-basin combinations. They are stored in a Pandas DataFrame named ```countrybasin_df``` together with their country-basin, basin and country ids. Each country-basin pair is an individual sub-model with separate deep copies of instantiated core model components. Country-basin pairs represent a geographical aspect of SOM analysis. Sub-division to smaller geographical regions is possible, but omitted from this version to keep it simple.

### Next development steps
- implementation of expert panel survey data distributions
- reading and processing ```pressStateInput.xlsx```
- implementation of model parts combining interactions and data: Pressure, State, Pressure-State
- implementing simple version of business logic
- module that implements distribution based calculation of business logic



