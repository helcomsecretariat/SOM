As outlined in the [File structure](../guide/file-structure.md) page, the tool consist of a few separate sub-modules that together comprise the complete SOM module. Each sub-module targets a specific part of the tools functionality, efficiently separating tasks into distinct sections.

## ``main.py``

This file manages the user interactions and calls to the core `som_app.py` module. If run directly, it will read the `config.toml` file and parse user specified arguments to change the config settings accordingly. Once done, it will run the `run()` method, which acts as the main SOM wrapper method. Here, the following occurs:

1. A directory for the indivudal simulation run logs is created in the same directory as the main script.
2. Output directories for the results are created.
3. The input data is loaded and processed.
4. The simulations are run by calling `run_sim()` for each individual simulation run. If the `parallell_processing` setting is set to `true`, the calculations will be run in parallell for faster processing. Each simulation's results are saved to a pickle file to conserve data structures.
5. The results are calculated from the congregation of all simulation round results.
6. The results are exported to the specified directory.
7. Plots are created to visualize the results.

## ``som_app.py``

This file acts as the core module of the package, containing the main calculations of the framework. The calculations called from `main.run_sim()` follow these steps:

1. `build_links()`

    1. Measure reductions for the current simulation run are calculated from the input data probability distributions.
    2. Activity contributions for the current simulation run are calculated from the input data probability distributions.
    3. Pressure contributions for the current simulation run are calculated from the input data probability distributions.
    4. Thresholds for the current simulation run are calculated from the input data probability distributions.

2. `build_scenario()` if `use_scenario` setting is set to `true`

    1. Activity contributions are adjusted by scenario multiplier
    2. Activity contributions are normalized so that for each pressure, the changes in contributions is reflected in the total sum of the contributions

3. `build_cases()`

    1. Cases are filtered to only include measures that have a documented effect in the input data
    2. Cases are exploded, replacing the placeholder '0' value with all relevant activities, pressures and states
    3. Cases are filtered to exclude activity-pressure-state combinations without associated measure reduction
    4. Duplicate measure-activity-pressure-state combinations are removed, leaving only rows with the highest coverage and implementation

4. `build_changes()`

    1. Activity contributions are normalized to make sure they do not exceed 100 %
        - Below 100 % is allowed as it is not guaranteed all contributions are known
    2. Pressure contributions are normalized to make sure they do not exceed 100 %
        - Below 100 % is allowed as it is not guaranteed all contributions are known
    3. Pressure level reductions
        1. Measure reductions are modified by coverage and implementation multipliers
        2. Measure reductions are modififed by overlapping measures
        3. Pressure levels are adjusted by activity contributions multiplied with measure reductions
        4. Activity contributions are adjusted by reductions and normalized
    4. Total pressure load reductions
        1. Straight to state measures
            1. Measure reductions are modified by coverage and implementation multipliers
            2. Measure reductions are modified by overlapping measures
            3. Total pressure load levels are adjusted by reductions
        2. Pressure level reductions
            1. Reduction is determined as 1 - pressure level reductions
            2. Reduction is adjusted by changes also in current pressure's subpressures
            3. Total pressure load levels are adjusted by calculated reductions
            4. Pressure contributions are adjusted by reductions and normalized

5. `build_results()`

    Individual simulation run results are allocated into arrays, from which means and standard errors are calculated. 

## ``som_tools.py``

This file handles the pre-processing of the legacy input data, to make sure it follows the format accepted by the methods in `som_app.py`. Each method in the file addresses one component of the input data. The accepted format of the legacy input data is detailed in [Input  data (legacy)](../guide/input-data-legacy.md).

## ``utilities.py``

This file contains small utility methods that are used throughout the several scripts in the SOM package. The main target functions of these fall into three categories:

- exception handling methods
- progress displaying methods
- probability distribution methods

## ``som_plots.py``

This file handles the optional output of the simulation results into plots to visualize them. The `build_display()` method makes the calls to the other methods to create each different type of plot. The following types of plots are created:

- Total pressure load levels (TPL level over state) for each case area
- Pressure levels for each case area
- Target threshold reductions compared to actual TPL reductions
- Total pressure load levels for each separate state for each case area (State pressures)
- Activity contributions for each case area
- Pressure contributions for each case area
- Measure reduction effects

