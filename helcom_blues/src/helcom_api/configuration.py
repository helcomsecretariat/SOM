"""
This module contains URLs to HELCOM databases.

general link is in form:
http://maps.helcom.fi/arcgis/rest/services/MADS

Notes:
'commercial or migratory fish' shp are not included
'introduction of non-indigenous species and translocations' commented out


Created on 2021.06.04. by Antti-Jussi Kieloaho
"""

from json import dumps

input_files = {
    'general_input': 'data/generalInput.xlsx',
    'measure_effect_input': 'data/measureEffInput.xlsx',
    'pressure_state_input': 'data/pressStateInput.xlsx'
} 

# GeoJSON RFC 7946
# bounding box as a GeoJSON form

model_domain_small = dumps({
    'type': 'Polygon',
    'coordinates': [[
        [5441000.0, 3395000.0],
        [5441000.0, 4813000.0],
        [4210000.0, 4813000.0],
        [4210000.0, 3395000.0],
        [5441000.0, 3395000.0]    
    ]],
    'crs': {
        'properties': {'name': 'EPSG:3035'}
    }
})

model_domain = dumps({
    'type': 'Polygon',
    'coordinates': [[
        [5591209.4265999998897314, 2945807.0],
        [5591209.4265999998897314, 5168051.0],
        [4204901.0156999994069338, 5168051.0],
        [4204901.0156999994069338, 2945807.0],
        [5591209.4265999998897314, 2945807.0]    
    ]],
    'crs': {
        'properties': {'name': 'EPSG:3035'}
    }
})
# 4204901.0156999994069338,2945807.0000000000000000
# 5591209.4265999998897314,5168051.0000000000000000


resolution = (-1000.0, 1000.0)

calculation_domains = {
    # Overlay intersection
    # domains: marine, terrestrial, north_sea, rest_of_world

    # borders_shp contains geometries and attributes of administrative boundaries
    # layer_shp contains geometries and attributes of  regions
    # borders_attrs contains attribute-value pairs needed to select right geometries
    # marine_attrs contains attributes that are reserved to calculation_domain after overlay

    'marine': {
        'administrative_shp': 'administrative boundaries',
        'administrative_attrs': [('Sea_area', 'EEZ'), ('Sea_area', 'TW')],
        'geographical_shp': 'sub-basins',
        'geographical_attrs': ['Country', 'HELCOM_ID', 'geometry']
       
    },
    'terrestrial': { 
        'administrative_shp': 'administrative boundaries', 
        'geographical_shp': 'specific load nitrogen',
        'administrative_attrs': [('Sea_area', 'Land')],
        'geographical_attrs': ['Country', 'SourceCode', 'SubBasin', 'geometry']
    },
    'north_sea': {
        'attributes': ['Country',]
    },
    'rest_of_world': {
        'attributes': ['Country',]
    },
}

# HELCOM MAPS API url
# service url is checked 13.9. 2021 by Antti-Jussi Kieloaho
service_url = 'https://maps.helcom.fi/arcgis/rest/services/MADS/tools/GPServer/getAndCompressDataSource/execute?id=&f=json'


# data layer list
name = 'name'  # data layer name
layer_id = 'id'  # data layer permanent id in database
file_name = 'file_name'  # path to data layer file (.shp or .tif file)
direct_link = 'direct_link'  # ArcGIS REST API link
comments = 'comments'  # comments about data layer
buffer = 'buffer'  # buffer layer aroung feature in meters [m]
summing_layers = 'summing_layers'  # The human activities were firts processed separately and then summed together
weight_factor = 'weight_factor'  # In the summing process of the human activity layers data layers have down-weight-factor
columns = 'columns'  # data columns that are important; if empty only geometry with binary data (1 if present, 0 if absent)
aggregation = 'aggregation'  # aggregation method of columns e.g. 'sum', 'average'


data_layers = [
    {
        name: 'catchments (JRC-CCM2)',
        layer_id: 'f0e7837c-8df2-4924-a675-dc221c2341b8',
        columns: [],
    },
    # Is following layer necessary?
    #{
    #    name: 'PLC subbasins',
    #    layer_id: '1456f8a5-72a2-4327-8894-31287086ebb5',
    #    columns: [],
    #},
    {
        name: 'sub-basins',
        layer_id: 'd4b6296c-fd19-462c-94d2-4c81b9313d77',
        columns: ['HELCOM_ID'],
    },
    # Is following layer necessary?
    #{
    #    name: 'Administrative boundaries',
    #    #layer_id: 'b58b0b56-1a98-41d9-9bf4-5dc76f5069d3',
    #    file_name: 'data/_ags_Administrative_boundaries/Administrative_boundaries.shp',
    #    columns: []
    #},
    {
        name: 'administrative boundaries',
        #layer_id: 'ae58c373-674c-45d1-be0f-1ff69a59f9ba',
        file_name: 'data/EEZ/EEZ_TW_Land_borders.shp',
        columns: []
    },
    {
        name: 'specific load nitrogen',
        layer_id: '20a44adc-8057-4fdb-9ce3-663095b4b967',
        comments: 'larger catchment units, small catchments are cut off',
        columns: []
    },
    {
        name: 'harbour porpoise distribution',
        layer_id: '0c359a75-bb20-4603-a447-fa9381c2b088',
        comments: 'will need to adjust cell category values, needs to be separated by management unit',
        columns: ['levelOfSpa']
    },
    {
        name: 'grey seal distribution',
        layer_id: '435ebf86-16e0-4bac-aac9-6055699d56a2',
        comments: 'need to consider adjusting cell category values',
        columns: ['levelOfSpa']
    },
    {
        name: 'ringed seal distribution',
        layer_id: 'ea7ee847-bf5d-4aa7-9a3e-340e40f7903b',
        comments: 'category values might be ok, needs to be separated by management unit',
        columns: ['levelOfSpa']
    },
    {
        name: 'harbour seal distribution',
        layer_id: 'f28fb72e-a579-4b55-83d9-8007206f8bb1',
        comments: 'need to consider adjusting cell category values, needs to be separated by management unit',
        columns: ['levelOfSpa']
    },
    {
        name: 'circalittoral hard substrate',
        layer_id: '5a6e7d7b-47d1-44fe-a70b-c0717bbd8566',
        comments: 'hard substrate epifauna dominated community',
        columns: [],
    },
    {
        name: 'infralittoral mixed substrate',
        layer_id: 'metadata/ca6796fb-54e9-475b-9dcc-53a41d366940',
        comments:'combined layer with infralittoral sand, mud, and mixed substrate; soft substrate vegitation dominated community; not a perfect representation but should be approximate',
        columns: []
    },
    {
        name: 'infralittoral mud',
        layer_id: 'a72559f4-bc2d-4882-bb1f-e0f005c4362b',
        comments: 'combined layer with infralittoral sand, mud, and mixed substrate; soft substrate vegitation dominated community; not a perfect representation but should be approximate',
        columns: [],
    },
    {
        name: 'infralittoral sand',
        layer_id: '6c68decf-3fb1-4138-9102-96d10495be64',
        comments: 'combined layer with infralittoral sand, mud, and mixed substrate; soft substrate vegitation dominated community; not a perfect representation but should be approximate',
        columns: [],
    },
    {
        name: 'somateria mollissima',
        layer_id: 'c3a216b5-9237-4ce8-a6fe-3aa115e83516',
        comments: 'common eider breeding season',
        columns: []
    },
    {
        name: 'physical disturbance',
        layer_id: '05e325f3-bc30-44a0-8f0b-995464011c82',
        comments: 'HAS METHODOLOGY FOR COMBINEING LAYERS',
        columns: ['levelOfSpa']
    },
    {
        name: 'physical loss',
        layer_id: 'ea0ef0fa-0517-40a9-866a-ce22b8948c88',
        comments: 'HAS METHODOLOGY FOR COMBINEING LAYERS',
        columns: []
    },
    {
        name: 'cables',
        layer_id: 'c0e73e71-cafb-4422-a3a3-115687fd5c49',
        buffer: 1000.0,
        weight_factor: 0.1,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: [],
    },
    {
        name: 'coastal defence',
        layer_id: '2d47c5ea-4590-465f-a462-60ef59d3d7d3',
        buffer: 500.0,
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: []
        #columns: ['Const_year']
    },
    {
        name: 'Deposit of dredged material areas 2011-2016',
        layer_id: '0eebe008-aed9-461a-8c85-e7b32bdab822',
        buffer: 500.0,
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['Amount2011', 'Amount2012', 'Amount2013', 'Amount2014', 'Amount2015', 'Amount2016'],
        aggregation: 'sum',
    },
    {
        name: 'Deposit of dredged material sites points 2011-2016',
        layer_id: '34bf84a1-1f04-40c7-99dd-9b9aff4523ea',
        buffer: 500.0,
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['Amount2011', 'Amount2012', 'Amount2013', 'Amount2014', 'Amount2015', 'Amount2016'],
        aggregation: 'sum'

    },
    {
        name: 'Dredging areas 2011-2016',
        layer_id: '2a0fbdfd-9aef-4d2e-9129-2d1cc3b4943b',
        buffer: 500.0,
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['TOT_M3']

    },
    {
        name: 'Dredging points 2011-2016',
        layer_id: 'bb6622b7-e5df-4637-8ebe-c2736e705a70',
        buffer: 500.0,
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: []
    },
    {
        name: 'Extraction of sand and gravel',
        layer_id: '683224c3-2fb9-4f2f-b748-bf5ad712d708',
        buffer: 500.0,
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['N2011TONNE', 'N2014TONNE', 'N2011M3', 'N2012M3', 'N2013M3', 'N2014M3', 'N2015M3'],
        aggregation: 'avg'
    },
    {
        name: 'Finfish mariculture',
        layer_id: '3cfa469a-6a78-4913-b82f-57fd0e7f4dc0',
        buffer: 1000.0,
        weight_factor: 0.6,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['P_AV_TON']
    },
    {
        name: 'Fishing intensity 2011-2016 average (subsurface swept area ratio)',
        layer_id: '8b7e8042-5911-4277-b1ea-d46f8fbdb413',
        weight_factor: 1.0,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['levelOfSpa']
    },
    {
        name: 'Furcellaria harvesting',
        layer_id: 'e72b634a-e4ca-4c1d-a3af-99a24aa10be8',
        weight_factor: 0.2,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['Total']
    },
    {
        name: 'Pipelines',
        layer_id: '5260249e-5850-431a-b130-3a096abac852',
        buffer: 300.0,
        weight_factor: 0.8,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: []
    },
    {
        name: 'Recreational boating and sports',
        layer_id: '8c30e828-1340-4162-b7f9-254586ae32b6',
        weight_factor: 0.2,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['levelOfSpa'],
    },
    {
        name: 'Shellfish mariculture areas',
        layer_id: '8fbf946a-18d2-4b24-98ad-1327ba2820a8',
        weight_factor: 0.6,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['AVG11_15'],
    },
    {
        name: 'Shellfish mariculture points',
        layer_id: '21efdecc-2fd7-4bfb-9832-68c995291bf7',
        weight_factor: 0.6,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: []
    },
    {
        name: 'Shipping density 2011-2015',
        layer_id: '1fd5298f-d3be-4c5f-bb05-3dc66a8eea6b',
        weight_factor: 0.8,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['levelOfSpa']
    },
    {
        name: 'Wind farms',
        layer_id: '80de3bc3-e3ec-474e-8ae8-4b29c205eb0a',
        buffer: 100.0,
        weight_factor: 0.2,
        summing_layers: ['physical loss', 'physical disturbance'],
        columns: ['Status']
    },
    {
        name: 'Bridges and other constructions',
        layer_id: '8a4b2ff4-30a1-430a-ba55-dfec57ebed97',
        columns: [],
    },
    {
        name: 'Harbours',
        layer_id: 'e2a3a104-a7aa-49c9-bc2c-47845b3a322f',
        columns: [],
    },
    {
        name: 'Land claim',
        layer_id: '79f6cb3c-b7f4-4ea7-8b36-59945602a221',
        columns: []
    },
    {
        name: 'Oil platforms',
        layer_id: 'fddc27ec-b6ae-407b-9dd8-f31a42d75412',
        columns: []
    },
    {
        name: 'Oil terminals',
        layer_id: '47120944-1b8e-4d47-832a-7f502d22ad88',
        columns: []
    },
    {
        name: 'Watercourse modification',
        layer_id: 'c1d21d0b-baf9-4429-86be-9e1570c574ef',
        columns: [],
    },
    {
        name: 'Marinas and leisure harbours',
    },
    {
        name: 'esturaries',
        layer_id: 'd027c58b-aceb-4b51-940d-73ea7c531251',
        comments: 'depends on how we handle new BSAP measures',
        columns: []
    },
    {
        name: 'HELCOM MPAs',
        layer_id: 'd27df8c0-de86-4d13-a06d-35a8f50b16fa',
        comments: 'may need specific polygons from layer',
        columns: ['MPA_status']
    },
    {
        name: 'Natura 2000 sites',
        layer_id: '47a94309-c72b-4a1a-8982-ed24ae829220',
        comments: 'may need specific polygons from layer',
        columns: []
    },
    {
        name: 'changes to hydrological conditions',
        layer_id: '50a19d2e-bc8a-40f3-a786-9e26274778a6',
        comments: "may be possible to project change based on layer's methods",
        columns: ['levelOfSpa']
    },
    {
        name: 'disturbance of species due to human presence',
        layer_id: '8e1389bc-ba28-480f-b7a6-f55fc43dde16',
        comments: 'may help limit areas where relevant measures can be useful',
        columns: ['levelOfSpa']
    },
    {
        name: 'extraction of seabirds',
        layer_id: 'e31c435c-0603-43d9-8d28-c61b04343e89',
        comments: "could be used to scale the impact of e.g. mpa's or other hunting restrictions",
        columns: ['levelOfSpa']
    },
    {
        name: 'input of continuous anthropogenic sound',
        layer_id: '8e73d7ab-d683-41f5-87ad-e97445e8eee5',
        comments: '125 Hz, could be used to scale the impact of measures',
        columns: ['levelOfSpa']
    },
    {
        name: 'input of impulsive anthropogenic sound',
        layer_id: '763adb5c-ece6-4cb7-8379-b9b1e65a6a83',
        comments: 'could be used to scale the impact of measures',
        columns: ['levelOfSpa']
    },
    # Is following layer necessary?
    #{
    #    name: 'introduction of non-indigenous species and translocations',
    #    layer_id: 'dff023a4-ad42-4a38-ade9-b4fd1194c79d',
    #    comments: 'rather old',
    #    columns: ['levelOfSpa']
    #},
]
