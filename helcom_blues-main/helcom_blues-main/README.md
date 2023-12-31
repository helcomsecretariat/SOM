# HELCOM_BLUES

Author: Antti-Jussi Kieloaho (antti-jussi.kieloaho@luke.fi; Natural Resources Institute Finland)

Project contains source codes of two separate programs, helcom-api and som. The first program is api consumer to retrieve and process spatial data in raster or vector form. The second set of source code is the first part of a program that processes, organises and calculates data for 'Sufficiency of Measures' analysis done at HELCOM.

## Installation

You can use helcom_api or som packages like set of libraries and call them, e.g., by using the jupyter notebooks provided (```src/helcom_api/sandbox.ipynb``` or ```src/som/sandbox.ipynb```) in Visual Studio Code IDE. You can install correct dependencies in virtual environment defined in ```pyproject.toml```: 

Python version has to be >= 3.9!

```
cd /path/to/helcom_blues
python -m venv .
source bin/activate
```

If you wish to have editable version of code use:

```
python -m pip install -e .  
```

If you want to use run only version:

```
python -m pip install .
```

## helcom_api

Contains HELCOM API consumer. Workflow of processing of spatial data fetched from HELCOM API is shown in Fig 1.

![image](docs/SOM-app-helcom-api.png)
Figure 1. Workflow of processing spatial data.

In ```src:helcom_api:data``` is directory ```EEZ``` containing topology corrected spatial shape files for exclusice economic zones and administrative boundaries. These are not available in HELCOM API, but given upon request.

In ```configuration.py``` there is meta data of spatial data and other information needed in processing.

In ```gis_tools.py``` contains io and processing functions of workflow.


## som (Sufficiency of Measures) model framework

In the first part development of SOM calculation platform, two of three of input data files are used for initialize and populate object model by programatically describing problem domain, in other words, interactions between societal, economic and envrionment in its geographcical context at Baltic Sea region. 

In construction of the the model three principles are followed when ever possible: 1. modularity, 2. object with single responsibility, and 3. clear interfaces. Even though, current version is a proof of concept of object model, care has been taken that it can extended in future versions, e.g., expected values extracted from survey data and description of the problem domain are single values with a range (min and max), not distributions, but implementation of distributions is already planned in future versions.

Input data consist three files generalInput.xlsx, measureEffInput.xslx, and pressStateInput.

generalInput.xlsx contains describtion of the problem domain:
- in sheet:ActMeas are governance issues, all rows are independent
    1. ID - ID of the case
    2. MT_ID - Measure type ID
    3. In_Activities - Relevant Activities, 0 means all relevant activities for Measure type in sheet:MT_to_A_to_P
    4. In_Pressure - Relevant Pressures, 0 means all relevant pressures for Measure type in sheet:MT_to_A_to_P
    5. In_State_components - Relevant States, 0 means all relevant states for Measure type in sheet:MT_to_A_to_P
    6. Multiplier_OL - Measure type multiplier
    7. B_ID - Basin IDs
    8. C_ID - Country IDs

- in sheet:ActPres are environmental issues, lists the relevant basins for each Activity-Pressure pairs
    1. Activity - Activity ID
    2. Pressure - Pressure ID
    3. Basin - Basin ID
    4. Ml# - MostLikely (ActivityPressure.AP_expected)
    5. Min# - Minimum end of range
    6. Max# - Maximum end of range
- in sheet:MT_to_A_to_P are linkages between measure types, activities, pressures and states


- measureEffInput.xslx: survey data on the effects of measures on activity-pressure pairs as surved by expert panels
- pressStateInput.xlsx

Model file structure (under ```src:som```):

- ```__main__```: calls facade functions that represents workflow steps. 
- ```configuration.py```: holds paths to input data files and other meta data about input files.
- ```som_app.py```: contains facade function that wraps underlying functions on thematic collections representing workflow steps.
- ```som_classes.py```: contains object definitions of core and secondary object layers.
- ```som_tools.py```: contains data reading and processing functions. 
- ```sandbox.ipynb```: jupyter notebook to run, test and check model objects.

## Object model of SOM

The model framework encapsulates problem domain and survey data as a layered structure as shown in Fig 2. 

![image](docs/SOM-app-class-diagram.png)
Figure 2. The class diagram of SOM

The layers consists of set of objects that describes basic elements of problem domain and their interactions. Each object contains appropriate data either from description of the problem domain, calculated from survey data, or calculated from all the previous. 

Object model describes the core functionality of SOM analysis, object oriented data structure resulted from survey data of expert panel together with description of problem domain. It concentrates to model Measure, Activity-Pressure, Activity and Pressure interactions and store approriate data. State, Pressure, and State-Pressure are included in object model but not implemented. 

In the initialization of object model following steps are taken

1. Process survey data from measure effect input and read general input
2. Build core object model and initialize core object instances
3. Build second object layer model and initialize seconf object layer instances
4. Post-process core objects organised in secondary object layer by storing data into them from general input

The data flow from input data to data structures containing object model is shown in Fig 3. Data flow diagram shows processes and data structures that are created upon a initalization.

![image](docs/SOM-app-data-flow-diagram.png)
Figure 3. The data flow diagram. 

After the initialization, altogether 41 object models are instantiated. They represents 41 country-basin combinations. They are stored in a form of a Pands DataFrame named ```countrybasin_df``` together with their country-basin, basin and country ids. Each country-basin pair are their own sub-model with a separate deep copies of instantiated core model components. Country-basin pair represents a geographical aspect of SOM analysis. Sub-division to smaller geographical regions is possible, but omitted from this version to keep it simple.

### Next development steps
- reading and processing in pressStateInput.xlsx
- implementation of model part combining interactions and data: Pressure, State, Pressure-State
- implementation of simple version of business logic
- module that implements distribution based calculation of business logic
- ```helcom_blues/som_tools.py:159: RuntimeWarning: divide by zero encountered in true_divide scaling_factor = np.divide(max_expected_value, max_effectivness)``` warning can be easily fixed by adding 'from np.finfo import eps' and adding eps to max_effectivness



