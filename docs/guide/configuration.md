The ```config.toml``` file can be edited in a text editor and follows the [TOML format](https://toml.io/en/). Please note that the options are case sensitive. 

- ```export_path```: Path of the output excel file
- ```use_scenario```: Should an activity development scenario be applied? (true/false)
- ```scenario```: Which activity development scenario to use
- ```use_random_seed```: Should a custom seed be used for random values? (true/false)
- ```random_seed```: Custom random seed to use
- ```simulations```: Number of simulations to run
- ```use_parallel_processing```: Should multiprocessing be used for faster calculations? (true/false)
- ```use_legacy_input_data```: Choose between legacy/new input data (true/false), see [Input data](input-data.md)
- ```input_data:path```: Path of the new input data excel file
- ```input_data_legacy:general_input```: Path to legacy input data general data excel file
- ```input_data_legacy:measure_effect_input```: Path to legacy input data measure effects data excel file
- ```input_data_legacy:pressure_state_input```: Path to legacy input data pressure-state links and thresholds data excel file
```input_data_legacy:general_input_sheets```: Links to the sheets in ```input_data_legacy:general_input``` in case of custom naming