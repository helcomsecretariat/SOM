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

from collections.abc import MutableMapping

from glob import glob
import shutil
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from geocube.api.core import make_geocube

import helcom_api.configuration as config

DataLayers = List[Dict[str, Any]]
LayerPaths = List[Dict[str, str]]
RenamedPaths = Dict[str, str]


class BijectiveMap(MutableMapping):
    """ Two-way hashmap for map lists to layers and wise verse
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


def rename_files(target_dirpath: str, ignore_pattern: str='^*') -> None:
    """ Return a dict and rename files. """

    root_dirpath = pathlib.Path(target_dirpath).expanduser()
    renamed_paths = {}

    for path in root_dirpath.rglob('*'):

        if not path.is_file() or path.match(ignore_pattern):
            continue

        # Parse new filename
        dir_path = path.parent
        suffix = path.suffix
        dir_name = dir_path.stem
        file_name = dir_name + suffix
        file_path = dir_path / file_name

        # Rename
        renamed_paths[str(path)] = file_path.name

        path.rename(file_path)


def get_layer_name(layer: Union[Dict[str, Any], str]) -> str:
    """
    Arguments:
        layer (dict): data layer item
    
    Returns:
        (str): data layer name
    """

    if isinstance(layer, dict):
        layer = layer['name']
    
    return layer.replace(' ', '_').replace('(', '').replace(')', '').lower()


def get_calculation_domain(layers, path_to_domains=None):
    """ Forms calculation domains based on countries, river catchments and sub-basins.

    Configuration of calculation_domains is in configuration.py
    Formed calculation domain as a .shp file is saved to data/domains

    Arguments:
        layers (dict): container that have shp-files

    Returns:
        path_to_domains (str): path to calculation domain layer
        calculation_domain_gdf (GeoDataFrame): calculation domain 

    """
        
    current_path = os.getcwd()

    if not path_to_domains:
        path_to_domains = os.path.join(current_path, 'data', 'domains')

    if not os.path.exists(path_to_domains):
        os.mkdir(path_to_domains)
        
    domain_layers = []
    
    for domain in config.calculation_domains:
        
        if domain == 'north_sea' or domain == 'rest_of_world':

            #attrs =  calculation_domains[domain]['attributes']
            #attrs.append('domain')

            #d = {'geometry': None, }

            #for col in attrs:
            #    d[col] = [domain]

            #gdf = gpd.GeoDataFrame(d)
            #domain_layers.append(gdf)
            
            continue

        borders_name = get_layer_name(config.calculation_domains[domain]['administrative_shp'])
        borders = layers['shp'][borders_name]
        borders_attrs = config.calculation_domains[domain]['administrative_attrs']

        layer_name = get_layer_name(config.calculation_domains[domain]['geographical_shp'])
        layer = layers['shp'][layer_name]

        for item in borders_attrs:
            # attributes from configuration.py
            # item is a tuple where first element is attribute and second element its value
            borders.loc[borders[item[0]] == item[1], 'domain'] = domain
            areas = borders.loc[borders['domain'] == domain]
            
        if 'Country' in areas:
            areas = areas.dissolve(by='Country').reset_index()
            
        domain_layer = gpd.overlay(df1=areas, df2=layer, how='intersection', keep_geom_type=True)
            
        attrs = config.calculation_domains[domain]['geographical_attrs']
        attrs.append('domain')
            
        domain_layer = domain_layer[attrs]
        domain_layers.append(domain_layer)
        
    # Combine separate marine and terrestrial calculation domains into one geodataframe (gdf)
    calculation_domain_gdf = pd.concat(domain_layers, ignore_index=True)
    calculation_domain_gdf['domain_index'] = calculation_domain_gdf.index

    # add rest_of_worl and north_sea into end with empty geometry

    path_to_domains = os.path.join(path_to_domains, 'calculation_domains.shp')
    calculation_domain_gdf.to_file(path_to_domains)

    return path_to_domains, calculation_domain_gdf


def _retrieve_from_helcom(layer_id: str, file_path: str, layer_name: str):
    """
    """

    if config.service_url:
        suburl = 'id='
        suburl_index = config.service_url.index(suburl) + len(suburl)
        prequest_url = config.service_url[:suburl_index] + layer_id + config.service_url[suburl_index:]
    else:
        raise ValueError("Service URL is not defined in configuration file.")

    # making prerequest to obtain direct download link to layer .zip file
    service_response = requests.get(prequest_url)
    service_response.raise_for_status()

    prequest_url = config.service_url[:suburl_index] + layer_id + config.service_url[suburl_index:]

    # making prerequest to obtain direct download link to layer .zip file
    service_response = requests.get(prequest_url)
    service_response.raise_for_status()

    # taking download link from prerequest response
    json = service_response.json()
    layer_url = json['results'][0]['value']['url']

    # creating path/to/layers/ for files if not existing
    os.makedirs(os.path.join(file_path, layer_name), exist_ok=True)

    # making request to obtain layer .zip file
    with requests.get(layer_url, allow_redirects=True) as r:
        r.raise_for_status()

        # extracting downloaded .zip files to path
        with zipfile.ZipFile(BytesIO(r.content)) as f:
            f.extractall(os.path.join(file_path, layer_name))


def preprocess_shp(layers, data_layers, raster_path: Optional[str] = None):
        """ Rasterizes shp files to be used in calculation.

        Step 1. adding buffer if buffer is determined
        Step 2. aggregation of specified attributes
        Step 3. rasterization of shp-attributes

        Note:
        at the moment only one kind of aggregation is possible per shp file
        
        Arguments: 
            raster_path (str): path to raster files 
        
        Returns:
            
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
                        resolution=config.resolution,
                        measurements=columns,
                        categorical_enums=categorical_enums,
                        fill=np.NaN,
                        geom=config.model_domain
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


def preprocess_files(data_layers: DataLayers, file_path: str, retrieve: bool = False) -> Tuple[LayerPaths, DataLayers]:
    """ 
    Preprocess map layer files and separates rasters from polygons. 

    This function loads layers either locally or by downloading them, 
    stores them in the designated location, and returns layer information.

    Arguments:
        data_layers (dict): layer information
        file_path (str): path/to/layer/files

    Returns:
        layer_paths (dict): paths of layer files
            'shp': polygon name: path/to/polygon.shp
            'tif': raster name: path/to/raster.tif
        layers (dict): layers
            'shp': polygon name: GeoDataFrame
            'tif': raster name: dataset
        retrieve (bool): fetch datalayers from HELCOM; default False.
    """

    current_path = os.getcwd()

    if file_path is None:
        file_path = os.path.join(current_path, 'data', 'layers')

    for layer in data_layers:
        layer_name = get_layer_name(layer)

        if 'id' in layer and retrieve:
            layer_id = layer['id'].split(os.path.sep)[-1]
            _retrieve_from_helcom(layer_id=layer_id, file_path=file_path, layer_name=layer_name)
        
        if 'file_name' in layer:

            # first, path is split to get file name 
            # second, file extension is split to get stem of file name
            # third, wild card is added into end
            file_name = layer['file_name'].split(os.path.sep)[-1]
            file_name = file_name.split('.')[0]
            file_name = file_name + '.*'
            
            # location of data layers locally
            path_to_data = layer['file_name'].rsplit(os.path.sep, 1)[0]
            path_to_layer = os.path.join(current_path + '/data/layers')
            
            layer_name = get_layer_name(layer)
            
            path_to_layer = os.path.join(path_to_layer, layer_name)
            
            file_list = glob(os.path.join(current_path, path_to_data, file_name))

            if not os.path.exists(path_to_layer):
                os.mkdir(path_to_layer)

            for f in file_list:
                f = f.split(os.path.sep)[-1]
                shutil.copy(src=os.path.join(current_path, path_to_data, f), dst=os.path.join(path_to_layer, f))

    rename_files(file_path)

    layer_paths = {
        'shp': {},
        'tif': {},
    }

    layers = {
        'shp': {},
        'tif': {}
    }


    for subdir, dirs, files in os.walk(file_path):
        for file_name in files:
            fp = subdir + os.path.sep + file_name

            if fp.endswith('.shp'):
                layer_paths['shp'][file_name.split('.')[0]] = fp
                geo_df = gpd.read_file(fp)
                layers['shp'][file_name.split('.')[0]] = geo_df

            elif fp.endswith('.tif'):
                layer_paths['tif'][file_name.split('.')[0]] = fp
                raster = rasterio.open(fp)
                layers['tif'][file_name.split('.')[0]] = raster


    # if both tif and shp are available, tif is prefered
    # shp-dicts are cleaned
    for key in layers['tif']:
        if key in layers['shp']:
            layers['shp'].pop(key)
            layer_paths['shp'].pop(key)


    path_to_domains = os.path.join(current_path, 'data', 'domains')
    path_to_domains, calculation_domain_gdf = get_calculation_domain(layers=layers, path_to_domains=path_to_domains)

    layer_paths['shp'] = path_to_domains
    layers['shp'].update({'calculation_domain': calculation_domain_gdf})

    return layer_paths, layers
