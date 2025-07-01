"""
Main script calling methods to do SOM calculations.
"""

# main package script

import som_app
import som_plots
from utilities import Timer, fail_with_message, display_progress

import toml
import os
import pandas as pd
import numpy as np
import sys
import copy
import shutil
import multiprocessing
import pickle


def read_config(config_file: str = 'config.toml'):
    """
    Reads configuration file.

    Arguments:
        config_file (str): path to configuration file, defaults to 'config.toml'.

    Returns:
        config (dict): configuration settings.
    """
    print('Reading configuration...')
    try:
        if not os.path.isfile(config_file): config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)
        with open(config_file, 'r') as f:
            config = toml.load(f)
    except Exception as e:
        fail_with_message('ERROR! Could not load config file!', e)
    return config


def run_sim(id: int, input_data: dict[str, pd.DataFrame], config: dict, out_path: str, log_path: str, progress, lock):
    """
    Runs a single simulation round.

    Arguments:
        id (int): Simulation round identifier.
        input_data (dict[str, DataFrame]): Input data used for calculations.
        config (dict): User configuration settings.
        out_path (str): Output path for results.
        log_path (str): Output path for log.
        progress (Namespace): multiprocessing.Manager.Namespace:

            - current (int): Current amount of finished simulations.
            - total (int): Total amount of simulations to calculate.
        
        lock (Lock): multiprocessing.Manager.Lock, used to manage concurrent processes updating progress.

    Returns:
        out (int): 0 (failure) | 1 (success)
    """
    log = open(log_path, 'w')
    
    try:

        print(f'sim = {id}', file=log)

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
        # export to pickle
        with open(out_path.replace('xlsx', 'pickle'), 'wb') as f:
            pickle.dump(data, f)

        with lock:
            progress.current += 1
            display_progress(progress.current / progress.total, text='\tProgress: ')

    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong during simulation! Check traceback.', e, file=log, do_not_exit=True)
        log.close()
        return 0
    
    log.close()
    return 1


def run(config: dict, skip_sim: bool = False):
    """
    Main function that loads input data and user configuration, 
    runs simulations and processes results.

    Arguments:
        config (dict): configuration settings.
        skip_sim (bool): toggle to skip SOM calculations and only process results.
    """
    # create log directory
    # NOTE! Existing logs are not deleted before new runs, only overwritten
    log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    #
    # setup paths
    #

    # main result directory
    export_path = os.path.realpath(config['export_path'])
    if not os.path.isdir(os.path.dirname(export_path)): export_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['export_path'])
    # individual simulation results directory
    sim_res_dir = os.path.join(os.path.dirname(export_path), 'sim_res')
    if os.path.exists(sim_res_dir):
        for f in [x for x in os.listdir(sim_res_dir) if x.endswith('.xlsx') or x.endswith('.pickle') and 'sim_res' in x]:
            if not skip_sim:
                os.remove(os.path.join(sim_res_dir, f))
    os.makedirs(sim_res_dir, exist_ok=True)
    # plot directory
    out_dir = os.path.join(os.path.dirname(export_path), 'output')
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    #
    # run simulations
    #

    # controlled randomness
    if config['use_random_seed']:
        print(f'Using random seed: {config["random_seed"]}')
        np.random.seed(config['random_seed'])

    # process survey data and read general input
    print('Loading input data...')
    try:
        input_data = som_app.build_input(config)
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing input data! Check traceback.', e)
    
    # run simulations and do calculations
    print('Running simulations...')
    if not skip_sim:
        cpu_count = multiprocessing.cpu_count()     # available cpu cores
        with multiprocessing.Manager() as manager:
            progress = manager.Namespace()
            progress.current = 0
            progress.total = config['simulations']
            lock = manager.Lock()
            display_progress(progress.current / progress.total, text='\tProgress: ')
            if config['use_parallel_processing']:   # parallell processing for faster computations
                with multiprocessing.Pool(processes=(min(cpu_count - 2, config['simulations']))) as pool:
                    jobs = [(i, input_data, config, os.path.join(sim_res_dir, f'sim_res_{i}.xlsx'), os.path.join(log_dir, f'log_{i}.txt'), progress, lock) for i in range(config['simulations'])]
                    pool.starmap(run_sim, jobs)
            else:   # single core solution
                for i in range(config['simulations']):
                    run_sim(i, input_data, config, os.path.join(sim_res_dir, f'sim_res_{i}.xlsx'), os.path.join(log_dir, f'log_{i}.txt'), progress, lock)
            display_progress(progress.current / progress.total, text='\tProgress: ')
    
    #
    # process results
    #

    print('\nProcessing results...')
    try:
        print('\tCalculating means and errors...')
        res = som_app.build_results(sim_res_dir, input_data)    # get condensed results from the individual simulation runs
        print('\tExporting results to excel...')
        som_app.export_results_to_excel(res, input_data, export_path)   # write results to file for human reading
        if config['create_plots']:
            print('\tProducing plots...')
            som_plots.build_display(res, input_data, out_dir, config['use_parallel_processing'], config['filter'])   # plots various results for visual interpretation
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing results! Check traceback.', e)

    return


if __name__ == "__main__":
    timer = Timer()     # start a timer to track computation time
    print('\nInitiating program...\n')

    config_file = 'config.toml'
    config = read_config(config_file)

    skip_sim = False

    # parse arguments
    if '--ui' in sys.argv:  # only if launched from UI
        # verify validity of sys.argv
        try:
            assert ('--input_data' in sys.argv) or ('--general_input' in sys.argv and '--measure_effects' in sys.argv and '--pressure_state' in sys.argv)
            assert ('--export_path' in sys.argv)
            assert ('--simulations' in sys.argv)
            if ('--use_scenario' in sys.argv): assert ('--scenario' in sys.argv)
            if ('--use_random_seed' in sys.argv): assert ('--random_seed' in sys.argv)
            if ('--link_mpas_to_subbasins' in sys.argv):
                for arg in ['--mpa_layer_path', '--mpa_layer_id_attribute', '--mpa_layer_name_attribute', 
                            '--mpa_layer_measure_attribute', '--mpa_layer_measure_delimiter', 
                            '--subbasin_layer_path', '--subbasin_layer_id_attribute']:
                    assert (arg in sys.argv)
        except Exception as e:
            print('ERROR! Empty parameter fields!')
            exit()
        # modify config
        config['use_legacy_input_data'] = True if '--legacy' in sys.argv else False
        config['use_scenario'] = True if '--use_scenario' in sys.argv else False
        config['use_random_seed'] = True if '--use_random_seed' in sys.argv else False
        config['use_parallel_processing'] = True if '--use_parallel_processing' in sys.argv else False
        config['link_mpas_to_subbasins'] = True if '--link_mpas_to_subbasins' in sys.argv else False
        config['create_plots'] = True if '--create_plots' in sys.argv else False
        for i in range(len(sys.argv)):
            if sys.argv[i] in ['--general_input']:
                config['input_data_legacy']['general_input'] = sys.argv[i+1]
            if sys.argv[i] in ['--measure_effects']:
                config['input_data_legacy']['measure_effect_input'] = sys.argv[i+1]
            if sys.argv[i] in ['--pressure_state']:
                config['input_data_legacy']['pressure_state_input'] = sys.argv[i+1]
            if sys.argv[i] in ['--input_data']:
                config['input_data']['path'] = sys.argv[i+1]
            if sys.argv[i] in ['--export_path']:
                config['export_path'] = sys.argv[i+1]
            if sys.argv[i] in ['--simulations']:
                config['simulations'] = int(sys.argv[i+1])
            if sys.argv[i] in ['--scenario']:
                config['scenario'] = sys.argv[i+1]
            if sys.argv[i] in ['--random_seed']:
                config['random_seed'] = int(sys.argv[i+1])
            if sys.argv[i] in ['--mpa_layer_path']:
                config['layers']['mpa']['path'] = sys.argv[i+1]
            if sys.argv[i] in ['--mpa_layer_id_attribute']:
                config['layers']['mpa']['id_attr'] = sys.argv[i+1]
            if sys.argv[i] in ['--mpa_layer_name_attribute']:
                config['layers']['mpa']['name_attr'] = sys.argv[i+1]
            if sys.argv[i] in ['--mpa_layer_measure_attribute']:
                config['layers']['mpa']['measure_attr'] = sys.argv[i+1]
            if sys.argv[i] in ['--mpa_layer_measure_delimiter']:
                config['layers']['mpa']['measure_delimiter'] = sys.argv[i+1]
            if sys.argv[i] in ['--subbasin_layer_path']:
                config['layers']['subbasin']['path'] = sys.argv[i+1]
            if sys.argv[i] in ['--subbasin_layer_id_attribute']:
                config['layers']['subbasin']['id_attr'] = sys.argv[i+1]
    
    # run analysis
    run(config, skip_sim)

    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')
