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
import shutil


def run(config_file: str = None, skip_sim: bool = False):
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
    # paths
    #
    export_path = os.path.realpath(config['export_path'])
    if not os.path.isdir(os.path.dirname(export_path)): export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['export_path'])
    sim_res_path = os.path.join(os.path.dirname(export_path), 'sim_res')
    if os.path.exists(sim_res_path):
        for f in [x for x in os.listdir(sim_res_path) if x.endswith('.xlsx') and 'sim_res' in x]:
            if not skip_sim:
                os.remove(os.path.join(sim_res_path, f))
    os.makedirs(sim_res_path, exist_ok=True)
    out_dir = os.path.join(os.path.dirname(export_path), 'output')
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    #
    # run simulations
    #
    if config['use_random_seed']:
        print(f'Using random seed: {config["random_seed"]}', file=log)
        np.random.seed(config['random_seed'])

    # Process survey data and read general input
    print('Loading input data...')
    try:
        print('Loading input data...', file=log)
        input_data = som_app.build_input(config)
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing input data! Check traceback.', e)
    
    print('Running simulations...')
    if not skip_sim:
        display_progress(0)
        for i in range(config['simulations']):
            try:
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
                temp_path = os.path.join(sim_res_path, f'sim_res_{i}.xlsx')
                with pd.ExcelWriter(temp_path) as writer:
                    data['pressure_levels'].to_excel(writer, sheet_name='PressureLevels', index=False)
                    data['total_pressure_load_levels'].to_excel(writer, sheet_name='TPLLevels', index=False)
                    data['total_pressure_load_reductions'].to_excel(writer, sheet_name='TPLReductions', index=False)
                    data['thresholds']['PR'].to_excel(writer, sheet_name='RequiredReductionsForGES', index=False)
                    data['measure_effects'].to_excel(writer, sheet_name='MeasureEffects', index=False)
                    data['activity_contributions'].to_excel(writer, sheet_name='ActivityContributions', index=False)
                    data['pressure_contributions'].to_excel(writer, sheet_name='PressureContributions', index=False)

                display_progress((i + 1) / config['simulations'])

            except Exception as e:
                fail_with_message(f'ERROR! Something went wrong (sim i = {i})! Check traceback.', e)
    
    #
    # process results
    #
    print('\nProcessing results...')
    try:
        print('Calculating means and errors...', file=log)
        res = som_app.build_results(sim_res_path, input_data)
        print('Producing plots...', file=log)
        som_app.build_display(res, input_data, out_dir)
        print('Exporting results to excel...')
        with pd.ExcelWriter(export_path) as writer:
            new_res = som_app.set_id_columns(res, input_data)
            for key in new_res:
                new_res[key].to_excel(writer, sheet_name=key, index=False)
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing results! Check traceback.', e)

    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')

    log.close()

    return

if __name__ == "__main__":
    config_file = None
    skip_sim = False
    for i in range(len(sys.argv)):
        if sys.argv[i] in ['-config', '-c']:
            config_file = sys.argv[i+1]
        if sys.argv[i] in ['-skip', '-s']:
            skip_sim = True
    run(config_file, skip_sim)

