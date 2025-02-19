"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

# main package script

import toml
import som.som_app as som_app
from utilities import Timer, exception_traceback, fail_with_message
import os
import pandas as pd
import numpy as np
import sys


def run(is_test: bool = False):

    timer = Timer()
    print('\nInitiating program.')

    #
    # read configuration file
    #
    try:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.toml')
        with open(config_file, 'r') as f:
            config = toml.load(f)
            # if test_run, swap input files
            if is_test:
                for key in config['input_data'].keys():
                    if key in config['test_data'].keys():
                        config['input_data'][key] = config['test_data'][key]
    except Exception as e:
        fail_with_message('ERROR! Could not load config file!', e)

    #
    # do stuff
    #
    try:
        if config['use_random_seed']:
            np.random.seed(config['random_seed'])
        # Process survey data and read general input
        data = som_app.process_input_data(config)
        # Create links between core components
        data = som_app.build_links(data)
        if config['use_scenario']:
            # Update activity contributions to scenario values
            data['activity_contributions'] = som_app.build_scenario(data, config['scenario'])
        # Create cases
        data = som_app.build_cases(data)
        # Run model
        data = som_app.build_changes(data)

        #
        # export results
        #
        filename = config['export_path']
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with pd.ExcelWriter(filename) as writer:
            data['pressure_levels'].to_excel(writer, sheet_name='PressureLevels', index=False)
            data['total_pressure_load_levels'].to_excel(writer, sheet_name='TPLLevels', index=False)
            data['total_pressure_load_reductions'].to_excel(writer, sheet_name='TPLReductions', index=False)
            data['thresholds']['PR'].to_excel(writer, sheet_name='RequiredReductionsForGES', index=False)

    except Exception as e:
        fail_with_message('ERROR! Something went wrong! Check traceback.', e)
    
    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')

    return

if __name__ == "__main__":
    is_test = '-test' in sys.argv or '-t' in sys.argv
    run(is_test=is_test)

