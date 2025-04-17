"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

from utilities import *

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import requests

import random


def create_cases(config: dict, data: dict[str, pd.DataFrame]):
    """
    Loads areas from helcom map and data service and creates cases dataframe
    """
    #
    # Get areas from MPA layer
    #

    service_url = config['layers']['service_url'] + config['layers']['area']['id'] + '/query'
    params = {
        'where': '1=1', 
        'outFields': '*', 
        'f': 'geoJSON'
    }
    r = requests.Request(method='GET', url=service_url, params=params).prepare().url
    gdf = gpd.read_file(r)

    #
    # replace this once actual layer structure is known
    #

    # create mock data
    gdf[config['layers']['area']['measure_attr']] = None
    gdf[config['layers']['area']['measure_attr']] = gdf[config['layers']['area']['measure_attr']].apply(lambda x: ';'.join(np.unique([str(random.randint(0, 9)) for i in range(10)])))

    # explode so there's only one measure per row
    gdf[config['layers']['area']['measure_attr']] = gdf[config['layers']['area']['measure_attr']].apply(lambda x: x.split(';'))
    gdf = gdf.explode(column=config['layers']['area']['measure_attr'])

    # create the cases dataframe for the input data
    cases = {
        'ID': np.arange(len(gdf)), 
        'measure': gdf[config['layers']['area']['measure_attr']], 
        'activity': np.zeros(len(gdf)), 
        'pressure': np.zeros(len(gdf)), 
        'state': np.zeros(len(gdf)), 
        'coverage': np.ones(len(gdf)), 
        'implementation': np.ones(len(gdf)), 
        'area_id': gdf[config['layers']['area']['id_attr']]
    }

    data['cases'] = pd.DataFrame(cases)

    return data


