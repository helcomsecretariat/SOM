"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

# main package script

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
        pickle.load(f)


def run():

    try:
        # check if processed input data already exists, otherwise process it
        data_path = 'data.p'
        links_path = 'links.p'
        # if os.path.exists(data_path) and os.path.exists(links_path):
        if False:
            print('heh')
            # load pickled data
            data = p_load(data_path)
            links = p_load(links_path)
        else:
            # Process survey data and read general input
            data = som_app.process_input_data()
            # Create links between core components
            links = som_app.build_links(data)
            # Create cases
            data['cases'] = som_app.build_cases(data['cases'], links)
            # save the data as pickle objects
            p_save(data, data_path)
            p_save(links, links_path)

        state_ges = som_app.simulate(data, links)

        print(state_ges['PR'])
    except Exception as e:
        exception_traceback(e)
    
    return

if __name__ == "__main__":
    run()

