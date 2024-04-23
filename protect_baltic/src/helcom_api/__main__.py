"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

import os
import helcom_api.gis_tools as gis_tools

def make_preprocessing(config_file, file_dir):
    """ wrapper function for preprocessing
    """

    config = gis_tools.read_config(config_file)

    layer_paths, layers = gis_tools.preprocess_files(
        config=config,
        file_dir=file_dir
    )

    raster_path, meta_info = gis_tools.preprocess_shp(
        config=config,
        layers=layers,
        raster_path=None
    )

    return layer_paths, raster_path, meta_info


if __name__ == '__main__':

    file_dir = None

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configuration.toml')
    layer_paths = make_preprocessing(config_file=config_file, file_dir=file_dir)


# EOF