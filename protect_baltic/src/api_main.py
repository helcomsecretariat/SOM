"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

from utilities import *
import api_tools
import som_app

import os
import toml

def main():

    timer = Timer()
    print('\nInitiating program...\n')

    #
    # read configuration file
    #
    try:
        config_file = 'api_config.toml'
        if not os.path.isfile(config_file): config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_file)
        with open(config_file, 'r') as f:
            config = toml.load(f)
    except Exception as e:
        fail_with_message('ERROR! Could not load config file!', e)

    #
    # get the input data
    #
    print('Loading input data...')
    try:
        input_data = som_app.build_input(config)
    except Exception as e:
        fail_with_message(f'ERROR! Something went wrong while processing input data! Check traceback.', e)

    #
    # load areas from layers and adjust area ids
    #

    input_data = api_tools.link_areas(config, input_data)

    print(f'\nProgram terminated successfully after {timer.get_hhmmss()}')

    return

if __name__ == '__main__':
    main()
    input('Press ENTER to exit.')
