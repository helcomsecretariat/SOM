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


def run():
    timer = Timer()

    #
    # read configuration file
    #
    try:
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.toml')
        with open(config_file, 'r') as f:
            config = toml.load(f)
        
        # convert sheet name string keys to integers in config
        config['measure_survey_sheets'] = {int(key): config['measure_survey_sheets'][key] for key in config['measure_survey_sheets']}
        config['pressure_survey_sheets'] = {int(key): config['pressure_survey_sheets'][key] for key in config['pressure_survey_sheets']}
    except:
        print('Could not load config file!')
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

        state_ges = som_app.simulate(data, links)

        print(state_ges['PR'])
    except Exception as e:
        exception_traceback(e)
    
    print(f'\nProgram terminated after {timer.get_hhmmss()}')

    return

if __name__ == "__main__":
    run()

