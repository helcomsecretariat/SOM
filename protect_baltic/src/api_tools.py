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
import os
import random


def link_areas(config: dict, data: dict[str, pd.DataFrame]):
    """
    Links MPAs to subbasin data
    """
    mpa_id = config['layers']['area']['id_attr']
    subbasin_id = 'SUB_ID'

    #
    # Get subbasin-country combinations
    #

    path = config['layers']['subbasins']
    if not os.path.isfile(path): path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    subbasins = gpd.read_file(path)

    subbasins[subbasin_id] = subbasins['OBJECTID']
    subbasins = subbasins.drop(columns=['OBJECTID', 'fid'])

    # fix geometries if needed
    subbasins['geometry'] = subbasins.geometry.make_valid()

    #
    # Get areas from MPA layer
    #

    if 'path' in config['layers']['area'] and config['layers']['area']['path'] != "":
        path = config['layers']['area']['path']
        if not os.path.isfile(path): path = os.path.join(os.path.dirname(os.path.realpath(__file__)), path)
    else:
        path = config['layers']['area']['url']
    mpa = gpd.read_file(path)

    # fix geometries if needed
    mpa['geometry'] = mpa.geometry.make_valid()

    #
    # Get measures in MPAs
    # NOTE: modify this once actual layer structure is known
    #

    # create mock data
    mpa[config['layers']['area']['measure_attr']] = None
    mpa[config['layers']['area']['measure_attr']] = mpa[config['layers']['area']['measure_attr']].apply(lambda x: config['layers']['area']['measure_delimiter'].join(np.unique([str(random.randint(0, 9)) for i in range(10)])))

    # explode so there's only one measure per row
    mpa[config['layers']['area']['measure_attr']] = mpa[config['layers']['area']['measure_attr']].apply(lambda x: x.split(config['layers']['area']['measure_delimiter']))
    measures = mpa.explode(column=config['layers']['area']['measure_attr'])

    #
    # identify links between mpas and subbasins
    #

    # ensure both layers have the same crs
    mpa = mpa.to_crs(subbasins.crs)

    mpa = mpa.reset_index(drop=True)
    subbasins = subbasins.reset_index(drop=True)

    # perform spatial join on intersections
    intersects = gpd.sjoin(
        mpa[[mpa_id, 'geometry']], 
        subbasins[[subbasin_id, 'geometry']], 
        how='left', 
        predicate='intersects'
    )
    # remove rows without intersect
    intersects = intersects.dropna(subset=[subbasin_id])
    # create new columns for mpa and subbasin geometries
    merged = intersects.merge(
        mpa[[mpa_id, 'geometry']].rename(columns={'geometry': 'geom_mpa'}), 
        on=mpa_id
    ).merge(
        subbasins[[subbasin_id, 'geometry']].rename(columns={'geometry': 'geom_subbasins'}), 
        on=subbasin_id
    )
    # calculate intersect area between mpa and subbasin for each intersection (row)
    merged['intersect_area'] = merged.apply(
        lambda x: x['geom_mpa'].intersection(x['geom_subbasins']).area, axis=1
    )
    # find subbasin with the highest intersect area for each mpa
    max_idx = (merged.loc[merged.groupby(mpa_id)['intersect_area'].idxmax()][[mpa_id, subbasin_id]].set_index(mpa_id))
    # map the highest intersect area subbasin to the mpa geodataframe
    mpa[subbasin_id] = mpa[mpa_id].map(max_idx[subbasin_id])
    links = mpa.loc[:, [mpa_id, subbasin_id]]

    #
    # change area ids to match MPAs
    #

    # create the cases dataframe for the input data
    cases = {
        'ID': np.arange(len(measures)), 
        'measure': measures[config['layers']['area']['measure_attr']], 
        'activity': np.zeros(len(measures)), 
        'pressure': np.zeros(len(measures)), 
        'state': np.zeros(len(measures)), 
        'coverage': np.ones(len(measures)), 
        'implementation': np.ones(len(measures)), 
        'area_id': measures[config['layers']['area']['id_attr']]
    }

    data['cases'] = pd.DataFrame(cases)

    # create the area dataframe
    areas = pd.DataFrame({
        'ID': mpa[mpa_id].unique()
    })
    areas['area'] = areas['ID'].apply(lambda x: mpa.loc[(mpa[mpa_id] == x), config['layers']['area']['name_attr']].values[0])
    data['area'] = areas

    for key in ['activity_contributions', 'pressure_contributions', 'thresholds']:
        df = data[key]
        df = df.rename(columns={'area_id': subbasin_id})
        merged = df.merge(links, on=subbasin_id, how='inner')
        merged = merged.rename(columns={mpa_id: 'area_id'})
        merged = merged.drop(columns=[subbasin_id])
        data[key] = merged

    return data
