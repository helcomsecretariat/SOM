
import helcom_api.gis_tools as gis_tools

def make_preprocessing(config_file, file_dir, retrieve=False):
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
        data_layers=config['data_layers'],
        raster_path=None,
    )

    return layer_paths, raster_path, meta_info


if __name__ == '__main__':

    file_dir = None

    config_file = 'protect_baltic/src/helcom_api/configuration.toml'
    layer_paths = make_preprocessing(config_file=config_file, file_dir=file_dir)


# EOF