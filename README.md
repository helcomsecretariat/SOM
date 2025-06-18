# Sufficiency and Efficiency of Measures model

Model developed at HELCOM to assess what kind of improvements in environmental state and pressures can be achieved with existing measures [(more info)](https://helcom.fi/baltic-sea-action-plan/som/).

This project continues on the work done on the SOM model in the [HELCOM BLUES](https://github.com/helcomsecretariat/SOM/tree/main/legacy/helcom_blues) project as part of the [HELCOM PROTECT BALTIC](https://protectbaltic.eu/) project. 

## Use

See the [wiki](https://helcomsecretariat.github.io/SOM/guide/using-the-tool).

## SOM

The scientific foundation of the SOM model framework can be found [here](https://helcom.fi/baltic-sea-action-plan/som/).

### Context

The model links measures, human activities, pressures, and environmental states together to assess the improvements that can be achieved in the pressures and states through the implementation of measures. This is further compared against set thresholds to observe whether the improvements are sufficient to achieve Good Environmental Status (GES). 

### Input

Information on the input data is explained in the [wiki](https://helcomsecretariat.github.io/SOM/guide/input-data).

### Model flow

The model links together the various inputs as shown in the diagram below.

![som-model-flowchart](som_model_flow.png)

The figure above details the links between each element of the model, both for the original model developed in the [ACTION](https://helcom.fi/helcom-at-work/projects/action/) project and further expanded on in the [BLUES](https://helcom.fi/helcom-at-work/projects/blues/) project. For each individual simulation run, these calculations occur:

1. An individual random sample is picked from each of the probability distributions representing the measure reductions, activity and pressure contributions, and GES thresholds.

2. Activity contributions are multiplied by their respective development factor for the chosen scenario.

3. Pressure levels are reduced by the reduction in its respective activity contributions (or directly through straight-to-pressure measures).
    - Reduction = Coverage * Implementation * Measure overlaps * Measure reduction * Activity contribution

4. Total pressure load levels on environmental states are reduced by the reduction in its respective pressure contributions (or directly through straight-to-state measures).
    - Reduction = Pressure level reduction * Pressure contribution

### Next development steps

The tool will implement the following aspects:

- Multiple simulations to assess uncertainty of results
- Incorporating GIS layers in determining cases and areas
- Result visualisation

Possible future development steps to implement (but are not yet due to lack of data):

- Ecosystem services & Benefits
- Impact on human well-being
- Incentives
- Drivers

## Current codebase

#### PROTECT BALTIC (ongoing)

[Project description](https://helcom.fi/helcom-at-work/projects/protect-baltic/)  
[Code](/src)

## Past projects

#### ACTION

[Project description](https://helcom.fi/helcom-at-work/projects/action/)  
[Code](/legacy/helcom_action)

#### BLUES

[Project description](https://helcom.fi/helcom-at-work/projects/blues/)  
[Code](/legacy/helcom_blues)
