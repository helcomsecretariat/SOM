"""
Created 20210914 by Antti-Jussi Kieloaho (Luke)
Modified 03/2024 by Camilo Hernández (HELCOM)

Module contains tools to handle input data consumed by SOM-app
"""

from typing import Any, Optional, List, Dict, Tuple, Union
import zipfile
import os
from io import BytesIO
import pathlib
import requests     # get()
import toml

from collections.abc import MutableMapping

from glob import glob
import shutil
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from geocube.api.core import make_geocube

# import helcom_api.configuration as config

DataLayers = List[Dict[str, Any]]
LayerPaths = List[Dict[str, str]]
RenamedPaths = Dict[str, str]


class BijectiveMap(MutableMapping):
    """
    Two-way hashmap for map lists to layers and wise verse
    """

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)

    def __getitem__(self, key):
        return self.mapping[key]

    def __delitem__(self, key):
        
        value = self[key]
        del self.mapping[key]
        
        self.pop(value, None)

    def __setitem__(self, key, value):
        
        if key in self:
            del self[self[key]]
        
        if value in self:
            del self[value]

        self.mapping[key] = value
        self.mapping[value] = key

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"


def read_config(config_file: str) -> Dict[str, Any]:
    """
    Reads given configuration file and returns content as structured dictionary.

    Arguments:
        config_file (str): path/to/file

    Returns:
        data (dict): configuration file content
    """
    try:
        with open(config_file, 'r') as f:
            data = toml.load(f)
        # convert data_layers from dict to list
        data['data_layers'] = {
            layer: { 
                data['layer_attributes'][attr]: data['data_layers'][layer][attr]
                for attr in data['data_layers'][layer]
            } for layer in data['data_layers']
        }
        data['data_layers'] = [data['data_layers'][layer] for layer in data['data_layers']]
        return data
    except:
        print(f'Could not load config file: {config_file}')
        return None


def rename_files(target_dirpath: str, ignore_pattern: str='^*') -> Dict[str, Any]:
    """
    Return a dict and rename files. Files are renamed to name of their directory.

    Arguments:
        target_dirpath (str): directory to search recursively for files to rename
        ignore_pattern (str): regex pattern to exclude files

    Returns:
        renamed_paths (dict): dictionary {original_path: renamed_file_name}
    """
    root_dirpath = pathlib.Path(target_dirpath).expanduser()
    renamed_paths = {}

    for path in root_dirpath.rglob('*'):    # go through all files in directory

        if not path.is_file() or path.match(ignore_pattern):
            continue    # skip path

        # Parse new filename
        dir_path = path.parent
        suffix = path.suffix
        dir_name = dir_path.stem
        file_name = dir_name + suffix
        file_path = dir_path / file_name

        # Rename
        path.rename(file_path)
        renamed_paths[str(path)] = file_path.name

    return renamed_paths


def get_layer_name(layer: Union[Dict[str, Any], str], name: str = None) -> str:
    """
    Arguments:
        layer (dict | str): data layer item
        name (str): key to look for in data layer
    
    Returns:
        layer_name (str): processed data layer name
    """
    if isinstance(layer, dict) and name is not None and name in layer:
        layer_name = layer[name]
    else:
        layer_name = layer  # layer is just a string
    return layer_name.replace(' ', '_').replace('(', '').replace(')', '').lower()


def get_calculation_domain(calculation_domains: Dict[str, Any], layers: Dict[str, dict], domain_dir: str):
    """
    Forms calculation domains based on countries, river catchments and sub-basins.

    Configuration of calculation_domains is in configuration file
    Formed calculation domain as a .shp file is saved to data/domains

    Arguments:
        calculation_domains (dict): calculation domains from config file
        layers (dict): container that have shp-files
        domain_dir (str): path to calculation domain layer directory

    Returns:
        path_to_domains (str): path to calculation domain layer
        calculation_domain_gdf (GeoDataFrame): calculation domain 
    """    
    domain_layers = []
    
    print(f'----------\nCalculating domains...')
    for domain in calculation_domains:

        print(f'domain: {domain}')
        
        if domain == 'north_sea' or domain == 'rest_of_world':
            #attrs =  calculation_domains[domain]['attributes']
            #attrs.append('domain')
            #d = {'geometry': None, }
            #for col in attrs:
            #    d[col] = [domain]
            #gdf = gpd.GeoDataFrame(d)
            #domain_layers.append(gdf)
            continue

        # find the borders of the domain
        borders_name = get_layer_name(calculation_domains[domain]['administrative_shp'])
        borders = layers['shp'][borders_name]   # access domain layer from layer dict
        borders_attrs = calculation_domains[domain]['administrative_attrs']

        # go through attributes defined in configuration file
        # item is a tuple where first element is attribute and second element its value
        for item in borders_attrs:
            # select column 'domain'
            # select the rows with item value as value in item column
            # set those rows to have domain name as value in 'domain' column
            # if 'domain' column does not exist, it is created at this point
            borders.loc[borders[item[0]] == item[1], 'domain'] = domain
        
        # select the rows changed above, i.e. where value in 'domain' column is domain name
        areas = borders.loc[borders['domain'] == domain]

        # if there are various entries per country, merge them to get one entry per country
        if 'Country' in areas:
            areas = areas.dissolve(by='Country').reset_index()

        # geographical boundaries
        layer_name = get_layer_name(calculation_domains[domain]['geographical_shp'])
        layer = layers['shp'][layer_name]

        # make sure coordinate system is the same
        if layer.crs != areas.crs:
            layer = layer.to_crs(areas.crs)
        
        # create intersection of administrative attributes and geographical attributes
        domain_layer = gpd.overlay(df1=areas, df2=layer, how='intersection', keep_geom_type=True)

        # these are the attributes to be used in the domain layer
        attrs = calculation_domains[domain]['geographical_attrs']
        attrs.append('domain')  # add 'domain' to attributes
        
        # select given attribute columns from domain layer (created if not existing)
        # add them to list of domain layers
        domain_layer = domain_layer[attrs]
        domain_layers.append(domain_layer)
    
    print(f'Domains calculated.\n----------')
        
    # Combine separate marine and terrestrial calculation domains into one geodataframe (gdf)
    print('Combining domains to GeoDataFrame...')
    calculation_domain_gdf = pd.concat(domain_layers, ignore_index=True)
    calculation_domain_gdf['domain_index'] = calculation_domain_gdf.index

    # add rest_of_world and north_sea into end with empty geometry

    print('Saving domain...')
    # path_to_domains = os.path.join(domain_dir, 'calculation_domains.shp')
    path_to_domains = os.path.join(domain_dir, 'calculation_domains.gpkg')
    calculation_domain_gdf.to_file(path_to_domains)

    print('----------')
    return path_to_domains, calculation_domain_gdf


def retrieve_from_helcom(config: Dict[str, Any], layer_id: str, file_dir: str):
    """
    Retrieves desired file from helcom servers
    """
    service_url = config['properties']['service_url']

    if 'service_url' in config['properties']:
        service_url = config['properties']['service_url']
        suburl = 'id='
        suburl_index = service_url.index(suburl) + len(suburl)
        prequest_url = service_url[:suburl_index] + layer_id + service_url[suburl_index:]
    else:
        raise ValueError("Service URL is not defined in configuration file.")

    # making prerequest to obtain direct download link to layer .zip file
    service_response = requests.get(prequest_url)
    service_response.raise_for_status()

    # taking download link from prerequest response
    json = service_response.json()
    layer_url = json['results'][0]['value']['url']

    # creating path/to/layers/ for files if not existing
    os.makedirs(file_dir, exist_ok=True)

    # making request to obtain layer .zip file
    with requests.get(layer_url, allow_redirects=True) as r:
        r.raise_for_status()

        # extracting downloaded .zip files to path
        with zipfile.ZipFile(BytesIO(r.content)) as f:
            f.extractall(file_dir)


def preprocess_shp(config: Dict[str, Any], layers: DataLayers, data_layers: Dict[str, Any], raster_path: Optional[str] = None) -> Tuple[str, dict]:
    """
    Rasterizes shp files to be used in calculation.

    Step 1. adding buffer if buffer is determined
    Step 2. aggregation of specified attributes
    Step 3. rasterization of shp-attributes

    Note:
    at the moment only one kind of aggregation is possible per shp file
    
    Arguments: 
        config (dict): configuration file
        layers (dict): data from layers
        data_layers (dict): config file data layer information
        raster_path (str): path to raster files 
    
    Returns:
        raster_path (str): 
        meta_info (dict): 
    """
    # creating metadata about rasterized data layers
    # saved as a json file
    meta_info = {}

    for item in data_layers:
        name = get_layer_name(item)
        if name in layers['shp']:
                meta_info[name] = item
            
    # if buffer in meta_info, buffer added
    # if aggregation in meta_info, specified columns aggregated and meta_info updated accordingly
    for layer, geodf in layers['shp'].items():

        if layer not in meta_info:
            meta_info[layer] = {}

        if layer == 'calculation_domain':
            meta_info[layer]['columns'] = ['domain_index']

        if 'columns' in meta_info[layer]:
            columns = meta_info[layer]['columns']
        
        # Step 1. adding buffer if buffer is determined
        if 'buffer' in meta_info[layer]:

            add_buffer = lambda x: x.geometry.buffer(meta_info[layer]['buffer'])

            geodf['geometry'] = geodf.apply(add_buffer, axis=1)
            layers['shp'][layer] = geodf

        # Step 2. aggregation of specified attributes  
        if 'aggregation' in meta_info[layer]:
                
            method = meta_info[layer]['aggregation']

            if method == 'sum':
                amount_sum = geodf[columns].sum(axis=1)
                geodf['sum'] = amount_sum
            
            elif method == 'avg':
                amount_avg = geodf[columns].mean(axis=1)
                geodf['avg'] = amount_avg
            
            # replacing specified columns by result of aggregation step by
            # discarding the rest!
            columns = [method]

        # Step 3. rasterization of shp-attributes

        # if attribute is categorical, it is transformed to numeric
        # categories are picked from data as categories are not included in data_layers.py
        # categorical enums are added to meta_info that is dumped to JSON file

        categorical_enums = {}
        if columns:
            for col in columns:
                if (geodf[col].dtypes.name == 'category' or geodf[col].dtypes.name == 'object'):
                    categories = geodf[col].drop_duplicates().values.tolist()
                    categorical_enums[col] = categories

        if not columns:
            geodf['binary'] = list(len(geodf) * [1])
            columns.append('binary')

        out_grid = make_geocube(
            vector_data=geodf,
            resolution=config['resolution'],
            measurements=columns,
            categorical_enums=categorical_enums,
            fill=np.NaN,
            geom=config['model_domain']
        )

        if not raster_path:
            cwd = os.getcwd()
            raster_path = os.path.join(cwd, 'rasterized/')
            os.makedirs(raster_path, exist_ok=True)

        raster_file_paths = {}
        for col in columns:
                
            raster_file_name = layer + '_' + col + '.tif'
            raster_file_path = os.path.join(raster_path, raster_file_name)

            out_grid[col].rio.to_raster(
                raster_path=raster_file_path,
                driver='GTiff',
            )

            raster_file_paths[raster_file_name] = raster_file_path

            layers['tif'][raster_file_name] = rasterio.open(raster_file_path)

        # update meta_info
        meta_info[layer]['paths_to_rasters'] = raster_file_paths
        if categorical_enums:
            meta_info[layer]['categorical_enums'] = categorical_enums
        
        meta_info[layer]['columns'] = columns

    with open(raster_path + '/preprocessed_data_layers.json', 'w') as f:
        json.dump(meta_info, f)

    return raster_path, meta_info


def preprocess_files(config: Dict[str, Any], file_dir: str = None) -> Tuple[LayerPaths, DataLayers]:
    """ 
    Preprocess map layer files and separates rasters from polygons. 

    This function loads layers either locally or by downloading them, 
    stores them in the designated location, and returns layer information.

    Arguments:
        config (dict): configuration file
        file_dir (str): path/to/layer/files

    Returns:
        layer_paths (dict): paths of layer files
            'shp': polygon name: path/to/polygon.shp
            'tif': raster name: path/to/raster.tif
        layers (dict): layers
            'shp': polygon name: GeoDataFrame
            'tif': raster name: dataset
    """
    current_path = os.getcwd()

    if not file_dir:
        file_dir = os.path.join(current_path, 'data', 'layers')
    os.makedirs(file_dir, exist_ok=True)

    # load all layers defined in config
    print(f'----------\nLoading layers...')
    for layer in config['data_layers']:
        print('-----')
        try:
            layer_name = get_layer_name(layer, config['layer_attributes']['name'])

            print(f'layer_name: {layer_name}')
            # check if layer already exists
            path_to_layer = os.path.join(file_dir, layer_name)  # layer location
            if os.path.exists(path_to_layer) and len(os.listdir(path_to_layer)) > 0:
                continue    # the path already exists with files inside
            else:
                os.makedirs(path_to_layer, exist_ok=True)   # create path if it doesn't exist
            
            # load locally
            if config['layer_attributes']['file_name'] in layer:

                file_path = layer[config['layer_attributes']['file_name']]
                if not os.path.exists(file_path):
                    print(f'File not found: {file_path}')
                    continue    # file was not found, skip layer
                file_name = pathlib.Path(file_path).resolve().stem   # get basename without extension
                file_name = file_name + '.*'    # add wildcard to capture all files later
                path_to_data = pathlib.Path(file_path).parent.resolve()     # location of source files
                file_list = glob(os.path.join(path_to_data, file_name)) # lists all files "file_name.*"

                # copy data files to layer path
                for f in file_list:
                    f = pathlib.Path(f).name
                    shutil.copy(src=os.path.join(path_to_data, f), dst=os.path.join(path_to_layer, f))
                print(f'Successfully loaded.')

            # load from helcom api
            elif config['layer_attributes']['layer_id'] in layer:

                layer_id = layer[config['layer_attributes']['layer_id']].split(os.path.sep)[-1]
                try:
                    retrieve_from_helcom(config=config, 
                                      layer_id=layer_id, 
                                      file_dir=os.path.join(file_dir, layer_name))
                except ValueError as e:
                    print(f'{e}')
                print(f'Successfully downloaded.')
        
        except Exception as e:
            print(f'Could not load layer:\n{e}')
    print(f'-----\nLayers loaded.\n----------')

    # rename files to layer name instead of layer id (or anything else)
    rename_files(file_dir)

    # dictionaries for storing layer data and paths
    layer_paths = { 'shp': {}, 'tif': {} }
    layers = { 'shp': {}, 'tif': {} }

    # walk through all layer files and add to dictionaries
    for subdir, dirs, files in os.walk(file_dir):
        for file_name in files:
            fp = subdir + os.path.sep + file_name
            if os.path.isfile(fp):
                # shape files
                if fp.endswith('.shp'):
                    layer_paths['shp'][pathlib.Path(file_name).stem] = fp   # add path
                    geo_df = gpd.read_file(fp)  # open and read file as GeoDataFrame
                    layers['shp'][pathlib.Path(file_name).stem] = geo_df    # add GeoDataFrame
                # tif files
                elif fp.endswith('.tif'):
                    layer_paths['tif'][pathlib.Path(file_name).stem] = fp   # add path
                    raster = rasterio.open(fp)  # open raster file as dataset
                    layers['tif'][pathlib.Path(file_name).stem] = raster    # add dataset reference

    # if both tif and shp are available, tif is preferred and shp-dicts are cleaned
    for key in layers['tif']:
        if key in layers['shp']:
            layers['shp'].pop(key)
            layer_paths['shp'].pop(key)

    # calculation domains
    domain_dir = os.path.join(current_path, 'data', 'domains')
    os.makedirs(domain_dir, exist_ok=True)  # create domain directory if it does not exist
    path_to_domains, calculation_domain_gdf = get_calculation_domain(calculation_domains=config['calculation_domains'], 
                                                                     layers=layers, 
                                                                     domain_dir=domain_dir)

    layer_paths['shp'] = path_to_domains
    layers['shp'].update({'calculation_domain': calculation_domain_gdf})

    return layer_paths, layers
