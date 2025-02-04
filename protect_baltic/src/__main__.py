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
from utilities import Timer, exception_traceback
import os
import pickle
import pandas as pd
import sys


def p_save(data: object, path: str):
    """
    Saves the given data as a pickle object
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def p_load(path: str):
    """
    Loads pickle data
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


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
        print('Could not load config file!')
        exception_traceback(e)
        return

    #
    # do stuff
    #
    try:
        data_path = 'data.p'
        links_path = 'links.p'
        if config['pickle'] and os.path.exists(data_path) and os.path.exists(links_path):
            # load pickled data
            data = p_load(data_path)
            links = p_load(links_path)
        else:
            # Process survey data and read general input
            data = som_app.process_input_data(config)
            # Create links between core components
            links = som_app.build_links(data)
            if config['use_scenario']:
                # Update activity contributions to scenario values
                data['activity_contributions'] = som_app.build_scenario(data, config['scenario'])
            # Create cases
            data['cases'] = som_app.build_cases(data['cases'], links)
            
        if config['pickle']:
            # save the data as pickle objects
            p_save(data, data_path)
            p_save(links, links_path)

        data = som_app.build_changes(data, links)

        #
        # export results
        #
        filename = config['export_path']
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        with pd.ExcelWriter(filename) as writer:
            data['pressure_levels'].to_excel(writer, sheet_name='PressureLevels', index=False)
            data['total_pressure_load_levels'].to_excel(writer, sheet_name='TotalPressureLoadLevels')
            data['state_ges']['PR'].to_excel(writer, sheet_name='GapGES')

    except Exception as e:
        exception_traceback(e)
    
    print(f'\nProgram terminated after {timer.get_hhmmss()}')

    return

if __name__ == "__main__":
    is_test = '-test' in sys.argv or '-t' in sys.argv
    run(is_test=is_test)

