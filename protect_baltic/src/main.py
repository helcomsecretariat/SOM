"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

# main package script

import toml
import som_app
import som_plots
from utilities import Timer, fail_with_message, display_progress
import os
import pandas as pd
import numpy as np
import sys
import copy
import shutil
import multiprocessing
import pickle

# splash screen logo
som_logo = r"""
   _____  ____  __  __ 
  / ____|/ __ \|  \/  |
 | (___ | |  | | \  / |
  \___ \| |  | | |\/| |
  ____) | |__| | |  | |
 |_____/ \____/|_|  |_|
                       
Copyright (c) 2025 HELCOM
"""

def run_sim(id: int, input_data: dict[str, pd.DataFrame], config: dict, out_path: str, log_path: str, progress, lock):
    """
    Runs a single simulation round

    Arguments:
        id (int): Simulation round identifier
        input_data (dict[str, DataFrame]): Input data used for calculations
        config (dict): User configuration settings
        out_path (str): Output path for results
        log_path (str): Output path for log
        progress (Namespace): multiprocessing.Manager.Namespace:
            current (int): Current amount of finished simulations
            total (int): Total amount of simulations to calculate
        lock (Lock): multiprocessing.Manager.Lock: 
            used to manage concurrent processes updating progress

    Returns:
        0 | 1: Failure | Success
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


def run(config_file: str = None, skip_sim: bool = False):
    """
    Main function that loads input data and user confirguration, 
    runs simulations and processes results.
    """

    timer = Timer()     # start a timer to track computation time
    print('\nInitiating program...\n')

    # create log directory
    # NOTE! Existing logs are not deleted before new runs, only overwritten
    log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
    os.makedirs(log_dir, exist_ok=True)

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
        print('\tProducing plots...')
        som_plots.build_display(res, input_data, out_dir, config['use_parallel_processing'], config['filter'])   # plots various results for visual interpretation
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing results! Check traceback.', e)

    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')

    return

if __name__ == "__main__":
    # multiprocessing.freeze_support()
    print(som_logo)
    config_file = None
    skip_sim = False
    for i in range(len(sys.argv)):
        if sys.argv[i] in ['-config', '-c']:
            config_file = sys.argv[i+1]
        if sys.argv[i] in ['-skip', '-s']:
            skip_sim = True
    run(config_file, skip_sim)

    input('Press ENTER to exit.')

