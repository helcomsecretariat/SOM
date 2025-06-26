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
- ```input_data_legacy:general_input_sheets```: Links to the sheets in ```input_data_legacy:general_input``` in case of custom naming

- ```link_mpas_to_subbasins```: Choose to link areas with measures to the areas in the input data using polygon layer overlaps (true/false)
- ```layers:subbasin:url```: url to subbasin polygon layer
- ```layers:subbasin:path```: path to subbasin polygon layer, overrides url
- ```layers:subbasin:id_attr```: id attribute field of subbasin polygon layer, must correspond to ```input_data:area:ID``` values
- ```layers:mpa:url```: url to MPA polygon layer
- ```layers:mpa:path```: path to MPA polygon layer, overrides url
- ```layers:mpa:measure_attr```: measure attribute field, all measures of the MPA joined together by ```layers:mpa:measure_delimiter```
- ```layers:mpa:id_attr```: id attribute field of MPA polygon layer
- ```layers:mpa:name_attr```: name attribute field of MPA polygon layer
- ```layers:mpa:measure_delimiter```: delimiter used to separate measures in ```layers:mpa:measure_attr```
