"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

# main package script

import toml
import som_app as som_app
from utilities import Timer, fail_with_message
import os
import pandas as pd
import numpy as np
import sys


def run(config_file: str = None):

    timer = Timer()
    print('\nInitiating program...\n')

    #
    # read configuration file
    #
    try:
        if not config_file: config_file = 'config.toml'
        if not os.path.isfile(config_file): config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)
        with open(config_file, 'r') as f:
            config = toml.load(f)
    except Exception as e:
        fail_with_message('ERROR! Could not load config file!', e)

    #
    # do stuff
    #
    try:
        if config['use_random_seed']:
            np.random.seed(config['random_seed'])
        
        # Process survey data and read general input
        print('Loading input data...')
        data = som_app.process_input_data(config)

        # Create links between core components
        print('Building links between Measures, Activities, Pressures and States...')
        data = som_app.build_links(data)

        if config['use_scenario']:
            # Update activity contributions to scenario values
            print('Applying activity development scenario...')
            data['activity_contributions'] = som_app.build_scenario(data, config['scenario'])
        
        # Create cases
        print('Building cases...')
        data = som_app.build_cases(data)

        # Run model
        print('Calculating changes in environment...')
        data = som_app.build_changes(data)

        #
        # export results
        #
        print('Exporting results...')
        filename = os.path.realpath(config['export_path'])
        if not os.path.isdir(os.path.dirname(filename)): filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['export_path'])
        if not os.path.exists(os.path.dirname(filename)): os.makedirs(os.path.dirname(filename), exist_ok=True)
        with pd.ExcelWriter(filename) as writer:
            data['pressure_levels'].to_excel(writer, sheet_name='PressureLevels', index=False)
            data['total_pressure_load_levels'].to_excel(writer, sheet_name='TPLLevels', index=False)
            data['total_pressure_load_reductions'].to_excel(writer, sheet_name='TPLReductions', index=False)
            data['thresholds']['PR'].to_excel(writer, sheet_name='RequiredReductionsForGES', index=False)
            data['measure_effects'].to_excel(writer, sheet_name='MeasureEffects', index=False)
            data['activity_contributions'].to_excel(writer, sheet_name='ActivityContributions', index=False)
            data['pressure_contributions'].to_excel(writer, sheet_name='PressureContributions', index=False)

    except Exception as e:
        fail_with_message('ERROR! Something went wrong! Check traceback.', e)
    
    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')

    return

if __name__ == "__main__":
    config_file = None
    for i in range(len(sys.argv)):
        if sys.argv[i] in ['-config', '-c']:
            config_file = sys.argv[i+1]
    run(config_file)

