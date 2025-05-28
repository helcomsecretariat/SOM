
from helcom_api.gis_tools import preprocess_files, preprocess_shp
from helcom_api.configuration import data_layers

def make_preproessing(file_path, retrieve=False):
    """ wrapper function for preprocessing
    """    

    layer_paths, layers = preprocess_files(
        data_layers=data_layers,
        file_path=file_path,
        retrieve=retrieve
    )

    raster_path, meta_info = preprocess_shp(
        layers=layers,
        data_layers=data_layers,
        raster_path=None,
    )

    return layer_paths, raster_path, meta_info


if __name__ == '__main__':

    file_path = '/Users/ajk/Repositories/helcom_blues/data/layers/'
    retrive = False

    layer_paths = make_preproessing(file_path=file_path, retrieve=retrive)
    

# EOF