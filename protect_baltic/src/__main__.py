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
from utilities import Timer, fail_with_message, display_progress
import os
import pandas as pd
import numpy as np
import sys
import copy


def run(config_file: str = None):
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log.txt')
    log = open(log_path, 'w')

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
    # run simulations
    #
    export_path = os.path.realpath(config['export_path'])
    if not os.path.isdir(os.path.dirname(export_path)): export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['export_path'])
    sim_res_path = os.path.join(os.path.dirname(export_path), 'sim_res')
    if not os.path.exists(sim_res_path): os.makedirs(sim_res_path, exist_ok=True)

    if config['use_random_seed']:
        print(f'Using random seed: {config["random_seed"]}', file=log)
        np.random.seed(config['random_seed'])

    # Process survey data and read general input
    try:
        print('Loading input data...', file=log)
        input_data = som_app.process_input_data(config)
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing input data! Check traceback.', e)
    
    print('Running simulations...')
    display_progress(0)
    for i in range(config['simulations']):
        try:
            display_progress((i + 1) / config['simulations'])
            print(f'sim = {i}', file=log)

            # Create links between core components
            print('\tBuilding links between Measures, Activities, Pressures and States...', file=log)
            data = som_app.build_links(copy.deepcopy(input_data))

            if config['use_scenario']:
                # Update activity contributions to scenario values
                print('\tApplying activity development scenario...', file=log)
                data['activity_contributions'] = som_app.build_scenario(data, config['scenario'])
            
            # Create cases
            print('\tBuilding cases...', file=log)
            data = som_app.build_cases(data)

            # Run model
            print('\tCalculating changes in environment...', file=log)
            data = som_app.build_changes(data)

            #
            # export results
            #
            print('\tExporting results...', file=log)
            temp_path = os.path.join(os.path.dirname(export_path), 'sim_res', f'sim_res_{i}.xlsx')
            with pd.ExcelWriter(temp_path) as writer:
                data['pressure_levels'].to_excel(writer, sheet_name='PressureLevels', index=False)
                data['total_pressure_load_levels'].to_excel(writer, sheet_name='TPLLevels', index=False)
                data['total_pressure_load_reductions'].to_excel(writer, sheet_name='TPLReductions', index=False)
                data['thresholds']['PR'].to_excel(writer, sheet_name='RequiredReductionsForGES', index=False)
                data['measure_effects'].to_excel(writer, sheet_name='MeasureEffects', index=False)
                data['activity_contributions'].to_excel(writer, sheet_name='ActivityContributions', index=False)
                data['pressure_contributions'].to_excel(writer, sheet_name='PressureContributions', index=False)

        except Exception as e:
            fail_with_message(f'ERROR! Something went wrong (sim i = {i})! Check traceback.', e)
    
    #
    # process results
    #
    print('\nProcessing results...')
    res = som_app.build_results(sim_res_path)

    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')

    log.close()

    return

if __name__ == "__main__":
    config_file = None
    for i in range(len(sys.argv)):
        if sys.argv[i] in ['-config', '-c']:
            config_file = sys.argv[i+1]
    run(config_file)

