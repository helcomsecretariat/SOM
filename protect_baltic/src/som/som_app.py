"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

from copy import deepcopy

import os
import numpy as np
import pandas as pd
import toml

from som.som_tools import read_survey_data, preprocess_measure_survey_data, process_measure_survey_data, preprocess_pressure_survey_data, process_pressure_survey_data
from som.som_tools import read_core_object_descriptions, read_domain_input, read_case_input, read_linkage_descriptions, read_postprocess_data
from som.som_classes import Measure, Activity, Pressure, ActivityPressure, State, CountryBasin, Case


def process_input_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in data and processes it in usable form.

    Returns:
        measure_survey_df (DataFrame): contains the measure survey data of expert panels
        pressure_survey_df (DataFrame): contains the pressure survey data of expert panels
        object_data (dict): contains following data: 'measure', 'activity', 'pressure', 'state', 'domain', and 'postprocessing'
    """
    # read configuration file
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configuration.toml')
    with open(config_file, 'r') as f:
        config = toml.load(f)
    
    # convert sheet name string keys to integers in config
    config['measure_survey_sheets'] = {int(key): config['measure_survey_sheets'][key] for key in config['measure_survey_sheets']}
    config['pressure_survey_sheets'] = {int(key): config['pressure_survey_sheets'][key] for key in config['pressure_survey_sheets']}

    #
    # measure survey data
    #

    # read survey data from excel file
    file_name = config['input_files']['measure_effect_input']
    mteq, measure_survey_data = read_survey_data(file_name, config['measure_survey_sheets'])

    # preprocess survey data
    measure_survey_df = preprocess_measure_survey_data(mteq, measure_survey_data)

    # process survey data 
    measure_survey_df = process_measure_survey_data(measure_survey_df)

    #
    # pressure survey data
    #

    # read survey data from excel file
    file_name = config['input_files']['pressure_state_input']
    psq, pressure_survey_data = read_survey_data(file_name, config['pressure_survey_sheets'])

    # preprocess survey data
    pressure_survey_df = preprocess_pressure_survey_data(psq, pressure_survey_data)

    # process survey data 
    pressure_survey_df = process_pressure_survey_data(pressure_survey_df)

    #
    # measure / pressure / activity / state links
    #

    # read core object descriptions
    # i.e. ids for measures, activities, pressures and states
    file_name = config['input_files']['general_input']
    id_sheets = config['general_input_sheets']['ID']
    object_data = read_core_object_descriptions(file_name=file_name, id_sheets=id_sheets)

    # read calculation domain descriptions
    # i.e. ids for countries, basins and percentage of basin area by each country
    file_name = config['input_files']['general_input']
    id_sheets = config['general_input_sheets']['domain']
    domain_data = read_domain_input(file_name=file_name, 
                                    id_sheets = id_sheets, 
                                    countries_exclude=config['domain_settings']['countries_exclude'], 
                                    basins_exclude=config['domain_settings']['basins_exclude'])

    # read case input
    # links between measures, activities, pressures, basins and countries
    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['case']
    case_data = read_case_input(file_name=file_name, sheet_name=sheet_name)

    # read linkage descriptions
    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['linkage']
    linkage_data = read_linkage_descriptions(file_name=file_name, sheet_name=sheet_name)

    # read postprocessing data
    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['postprocess']
    postprocess_data = read_postprocess_data(file_name=file_name, sheet_name=sheet_name)

    object_data.update({
        'domain': domain_data,
        'cases': case_data,
        'linkages': linkage_data,
        'postprocessing': postprocess_data,
        })

    return measure_survey_df, pressure_survey_data, object_data


def build_core_object_model(msdf: pd.DataFrame, psdf: pd.DataFrame, object_data: dict[str, dict]) -> pd.DataFrame:
    """
    Builds and initializes core object model.

    Arguments:
        msdf (DataFrame): processed measure survey results
        psdf (DataFrame): processed pressure survey results
        object_data (dict): dictionary that contains following data: measure, activity, pressure, and state

    Returns:
        measure_df (DataFrame): 
    """

    # verify that there are no duplicate links
    assert len(msdf[msdf.duplicated(['measure', 'activity', 'pressure', 'state'])]) == 0

    #
    # Create activity objects
    #
    activities = {}
    activity_ids = msdf['activity'].unique()  # find unique activity ids
    for id in activity_ids:
        if id == 0:
            continue    # skip 0 index activities
        # divide id by multiplier to get actual id
        name = object_data['activity'].loc[object_data['activity']['ID']==int(id/10000)]['activity'].values[0]
        a = Activity(name=name, id=id)
        activities.update({id: a})

    #
    # Create pressure objects
    #
    pressures = {}
    pressure_ids = msdf['pressure'].unique()  # find unique pressure ids
    for id in pressure_ids:
        if id == 0:
            continue    # skip 0 index pressures
        name = object_data['pressure'].loc[object_data['pressure']['ID']==int(id)]['pressure'].values[0]
        p = Pressure(name=name, id=id)
        pressures.update({id: p})

    #
    # Create state objects
    #
    states = {}
    state_ids = msdf['state'].unique()  # find unique state ids
    for id in state_ids:
        if id == 0:
            continue    # skip 0 index states
        name = object_data['state'].loc[object_data['state']['ID']==int(id)]['state'].values[0]
        s = State(name=name, id=id)
        states.update({id: s})

    #
    # Create measure objects
    #
    measures = {}
    measure_ids = msdf['measure'].unique()     # find unique measure ids
    for id in measure_ids:
        if id == 0:
            continue    # skip 0 index measures
        name = object_data['measure'].loc[object_data['measure']['ID']==int(id/10000)]['measure'].values[0]
        m = Measure(name=name, id=id)
        measures.update({id: m})

    #
    # Create activity-pressure links
    #
    activitypressure_instances = {}
    for num in msdf['measure'].index:  # for every measure in the survey data

        # get the ids of the row
        measure_id = msdf['measure'].loc[num]
        activity_id = msdf['activity'].loc[num]
        pressure_id = msdf['pressure'].loc[num]
        # create the link to the measure
        if activity_id == 0 or pressure_id == 0:    # if the measure affects a state
            state_id = msdf['state'].loc[num]
            if state_id == 0:
                continue    # skip 0 index states
            measures[measure_id].add_state(states[state_id])
        else:
            ap = ActivityPressure(activity=activities[activity_id], pressure=pressures[pressure_id])
            if ap.id not in list(activitypressure_instances.keys()):
                activitypressure_instances.update({ap.id: ap})
            # link to ap from dict, so that there is only one object being linked
            measures[measure_id].activity_pressure = activitypressure_instances[ap.id]
    
        # set probability distribution
        measures[measure_id].expected_distribution = msdf['cumulative probability'].loc[num]

    # Rearrange initialized core objects in DataFrame
    # and make linkages between core objects
    measure_df = pd.DataFrame.from_dict({
        'instance': measures.values()
    })

    # ID:s are calculated so that they can be tracked
    measure_df['measure id'] = [int(x.id / 10000) * 10000 for x in measure_df['instance']]
    measure_df['activity-pressure id'] = [x.activity_pressure.id if x.activity_pressure != None else np.nan for x in measure_df['instance']]
    measure_df['activity id'] = [int(x.activity_pressure.id / 10000) * 10000 if x.activity_pressure != None else np.nan for x in measure_df['instance']]
    measure_df['pressure id'] = measure_df['activity-pressure id'] - measure_df['activity id']
    
    measure_df['activity id'] = [x.activity_pressure.activity.id if x.activity_pressure != None else np.nan for x in measure_df['instance']]
    measure_df['pressure id'] = [x.activity_pressure.pressure.id if x.activity_pressure != None else np.nan for x in measure_df['instance']]

    # Going through core objects and setting initial values
    # ActPres sheet values are still missing!

    for num in measure_df.index:

        m = measure_df.loc[num, 'instance']

        # measures have activity-pressure pairs or states
        if m.states:
            continue

        # TODO: the section below is wrong, go through Pressure and ActivityPressure classes and fix

        # setting expected value first on measure instance
        # multiplicating on activity-pressure instance based on measure instance
        # setting pressure reduction value in pressure instance  
        elif m.activity_pressure:
            ap = m.activity_pressure
            p = ap.pressure

            ap.expected = ap.expected * m.effect
            p.pressure_reduction = m.effect

        else:
            raise AttributeError("Measure instance is missing Activity-Pressure pair or State instances.")

    print(measure_df)
    exit()

    return measure_df


def build_second_object_layer(measure_df, object_data):
    """
    CountryBasins have their own set of Core objects. Core objects are trimmed to Cases

    - go through each row (raw case) in ActMeas sheet
    -> each case have basin-country combination
    -> each case have measure that is applied to basin-country combination
    ->fetch correct Measure instances
    ->fetch correct CountryBasin instances
    
    - initialize Case objects
    
    -> set correct Measure instance to CountryBasin
    -> set correct Measure instance to Case
    -> set Case instance to CountryBasin

    Note: One Measure is applied only once to one CountryBasin
    Note: case_id = cases[ID] * 100'000'000 + measure_id

    Arguments:
        measure_df (DataFrame): contains core object instances
        object_data (dict): dictionary that contains following data: domain
    """
    countries = object_data['domain']['country']
    basins = object_data['domain']['basin']
    countries_by_basins = object_data['domain']['countries_by_basins']

    instances = {}

    for country in countries['country']:
        country_id = countries[countries['country'] == country].index[0]
    
        for basin in basins['basin']:
            basin_id = basins[basins['basin'] == basin].index[0]
       
            basin_fraction = countries_by_basins.loc[(countries_by_basins.index == country_id), basin_id].values[0]

            if basin_fraction <= 0:
                continue

            countrybasin_id = basin_id * 1000 + country_id
            countrybasin_name = f"{country} ({country_id}) and {basin} ({basin_id})"

            cb = CountryBasin(id=countrybasin_id, name=countrybasin_name)
            cb.basin_fraction = basin_fraction

            instances.update({countrybasin_id: cb})

    # Rearrange CountryBasin objects to DataFrame
    countrybasin_df = pd.DataFrame.from_dict({
        'instance': instances.values()
    })

    countrybasin_df['country-basin id'] = [x.id for x in countrybasin_df['instance']]
    countrybasin_df['basin id'] = [int(x.id / 1000) * 1000 for x in countrybasin_df['instance']]
    countrybasin_df['country id'] = countrybasin_df['country-basin id'] - countrybasin_df['basin id']

    cases = object_data['cases']
    cases_num = cases.loc[:, 'ID'].unique()

    linkages = object_data['linkages']

    for c_num in cases_num:

        # In each case there is only one measure type
        measure_num = cases['MT_ID'].loc[cases['ID'] == c_num].unique()[0]
        
        # choose measures from a dataframe
        measures = measure_df.loc[measure_df['measure id'] == measure_num * 10000]

        countrybasins = cases['countrybasin_id'].loc[cases['ID'] == c_num]

        for cb_id in countrybasins.values:

            # This takes out special cases that should be handle separately
            if cb_id < 1000:
                continue

            # Skipping those countrybasin_id:s that does not exist
            if not countrybasin_df['country-basin id'].isin([cb_id]).any():
                continue

            # Taking countrybasin instance
            cb = countrybasin_df.loc[countrybasin_df['country-basin id'] == cb_id, 'instance']

            if len(cb) > 1:
                raise ValueError("Found more than one country-basin pair!")
                
            cb = cb.values[0]
            
            # Add measure with right activity-pressure association to list in countrybasin instance
            # take a measure from linkages
            # fetch relevant activities from linkages
            # fetch relevant pressures from linkages
                
            # Now adds all measures with activity-pressure pairs

            case_id = c_num * 10000
            case = Case(id=case_id)

            for measure in measures['instance']:
                m_id = measure.id
                m_num = int(m_id / 10000)
                
                relevants = linkages[linkages['MT'] == m_num]

                a_num = relevants['Activities']
                p_num = relevants['Pressure']

                if ~a_num.isna().any() or ~p_num.isna().any():
                    ap_id = a_num.astype(int) * 10000 + p_num.astype(int)
                    
                    for id in ap_id:

                        if id == measure.activity_pressure.id:

                            # each countrybasin have they own measure that might have different activity-pressure due to 
                            # differences in biogeochemical or physical conditions
                            measure_copy = deepcopy(measure)

                            cb.measures = measure_copy
                            case.measures = measure_copy

            cb.cases = case

    return countrybasin_df


def postprocess_object_layers(countrybasin_df, object_data):
    """
    Post-process core objects based on geographical position (country-basin objects)

    Arguments:
        countrybasin_df (DataFrame): contains country-basin instances
        object_data (dict): dictionary that contains following data: postprocessing
    """

    act_to_press = object_data['postprocessing']
    
    # Prepare countrybasin_df
    # each row has individual activity-pressure pairs arranged by country-basins.
    countrybasin_df['activity-pressure id'] = [[int(m.activity_pressure.id) for m in cb.measures] for cb in countrybasin_df['instance'].values]
    countrybasin_df = countrybasin_df.explode('activity-pressure id')
    countrybasin_df['activity id'] = (countrybasin_df['activity-pressure id'] / 10000).astype('int') * 10000
    countrybasin_df['activity'] = (countrybasin_df['activity-pressure id'] / 10000).astype('int')
    countrybasin_df['pressure'] = countrybasin_df['activity-pressure id'] - countrybasin_df['activity id']


    # Go through all basin values in act_to_press
    for i, basin in enumerate(act_to_press['Basins'].values):

        # find activity and pressure in that act_to_press row of basin
        activity = act_to_press['Activity'].iloc[i]
        pressure = act_to_press['Pressure'].iloc[i]

        ap_id = activity * 10000 + pressure

        expected = np.array(act_to_press['expected'].iloc[i]) / 100.0
        minimum = np.array(act_to_press['minimum'].iloc[i]) / 100.0
        maximum = np.array(act_to_press['maximum'].iloc[i]) / 100.0

        # case if applied to all basins
        if basin == '0':

            # find country-basin instances from countrybasin_df 
            # containing correct activity-pressure pair from act_to_press
            instances = countrybasin_df.loc[(countrybasin_df['activity'] == activity) & (countrybasin_df['pressure'] == pressure), 'instance']

            # all found basin instances
            for cb in instances:

                # measures that are in the specific country-basin instances
                for m in cb.measures:

                    # if measure is linked to right activity-pressure pair, set values act_to_press
                    if m.activity_pressure.id == ap_id:

                        m.activity_pressure.expected = expected
                        m.activity_pressure.min_expected = minimum
                        m.activity_pressure.max_expected = maximum

        # case if applied only one basin
        else:

            # find country-basin instances from countrybasin_df 
            # containing correct activity-pressure pair from act_to_press
            instances = countrybasin_df.loc[
                (countrybasin_df['activity'] == activity) &
                (countrybasin_df['pressure'] == pressure) & 
                (countrybasin_df['basin id'] == int(basin)*1000), 'instance']

            # all found basin instances
            for cb in instances:
                
                # measures that are in the specific country-basin instances
                for m in cb.measures:

                    # if measure is linked to right activity-pressure pair, set values from act_to_press
                    if m.activity_pressure.id == ap_id:

                        m.activity_pressure.expected = expected
                        m.activity_pressure.min_expected = minimum
                        m.activity_pressure.max_expected = maximum

    countrybasin_df = countrybasin_df.loc[:, ['instance', 'country-basin id', 'basin id', 'country id']].drop_duplicates()

    return countrybasin_df

#EOF