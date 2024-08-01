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
from som.som_tools import get_expert_ids, get_prob_dist
from som.som_classes import Measure, Activity, Pressure, ActivityPressure, State, CountryBasin, Case


def process_input_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in data and processes it in usable form.

    Returns:
        measure_survey_df (DataFrame): contains the survey data of expert panels
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
    file_name = config['input_files']['general_input']
    object_data = read_core_object_descriptions(file_name=file_name)

    # read calculation domain descriptions
    file_name = config['input_files']['general_input']
    domain_data = read_domain_input(file_name=file_name, 
                                    countries_exclude=config['domain_settings']['countries_exclude'], 
                                    basins_exclude=config['domain_settings']['basins_exclude'])

    # read case input 
    file_name = config['input_files']['general_input']
    case_data = read_case_input(file_name=file_name)

    # read linkage descriptions
    file_name = config['input_files']['general_input']
    linkage_data = read_linkage_descriptions(file_name=file_name)

    # read postprocessing data
    file_name = config['input_files']['general_input']
    postprocess_data = read_postprocess_data(file_name=file_name)

    object_data.update({
        'domain': domain_data,
        'cases': case_data,
        'linkages': linkage_data,
        'postprocessing': postprocess_data,
        })

    return measure_survey_df, object_data


def build_core_object_model(survey_df, object_data) -> pd.DataFrame:
    """
    Builds and initializes core object model.

    Arguments:
        survey_df (DataFrame): processed survey results
        object_data (dict): dictionary that contains following data: measure, activity, pressure, and state

    Returns:
        measure_df (DataFrame): 
    """
    # select measures, activities, pressures and states
    measures = survey_df['measure'].loc[(survey_df['title'] == 'expected value')]
    activities = survey_df['activity'].loc[(survey_df['title'] == 'expected value')]
    pressures = survey_df['pressure'].loc[(survey_df['title'] == 'expected value')]
    states = survey_df['state'].loc[(survey_df['title'] == 'expected value')]
    # select aggregated expected values and variances
    expecteds = survey_df['aggregated'].loc[(survey_df['title'] == 'expected value')]
    uncertainties = survey_df['aggregated'].loc[(survey_df['title'] == 'variance')]
    # select individual expert expected values, variances and boundaries
    expert_ids = get_expert_ids(survey_df)
    expert_expecteds = survey_df[expert_ids].loc[(survey_df['title'] == 'expected value')]
    expert_uncertainties = survey_df[expert_ids].loc[(survey_df['title'] == 'variance')]
    expert_lower_boundaries = survey_df[expert_ids].loc[(survey_df['title'] == 'effectivness lower')]
    expert_upper_boundaries = survey_df[expert_ids].loc[(survey_df['title'] == 'effectivness upper')]
    expert_weights = survey_df.loc[survey_df['title'] == 'expert weights', np.insert(expert_ids, 0, 'block')]
    measures_blocks = survey_df.loc[(survey_df['title'] == 'expected value'), ['measure', 'block']]

    # find unique ids
    activity_ids = activities.unique()
    pressure_ids = pressures.unique()

    activity_instances = {}
    pressure_instances = {}

    for id in activity_ids: # for each activity

        if id == 0:
            continue    # skip 0 index activities

        name = object_data['activity'][id/10000]    # activity name, divide by multiplier to get actual id
        a = Activity(id=id, name=name)  # create Activity object
        activity_instances.update({id: a})  # add activity to dictionary

    for id in pressure_ids: # for each pressure

        if id == 0:
            continue    # skip 0 index pressures
        
        name = object_data['pressure'][id]  # pressure name
        p = Pressure(id=id, name=name)  # create Pressure object
        pressure_instances.update({id: p})  # add pressure to dictionary

    measure_instances = {}
    activitypressure_instances = {}

    for num in measures.index:  # for every measure row (not unique)

        # instantiate Measure
        measure_id = measures.loc[num]  # find all occurences of that measure
        measure_name = object_data['measure'][int(measure_id/10000)]    # measure name
        m = Measure(id=measure_id, name=measure_name)   # create Measure object
        measure_instances.update({measure_id: m})   # add measure to dictionary

        # instantiate only State
        # Activity and Pressure have been instantiated above
        activity_id = activities.loc[num]
        pressure_id = pressures.loc[num]

        if int(activity_id) == 0 or pressure_id == 0:   # if the measure affects all activities or pressures
            state_id = states.loc[num]  # select state id list from 
            
            s_instances = []
            if isinstance(state_id, list):  # if the state_id is a list

                for id in state_id:
                    if id == '' or id == np.nan:
                        continue  # input files have inconsistencies

                    state_name = object_data['state'][int(id)]  # state name
                    s = State(id=id, name=state_name)   # create State object
                    s_instances.append(s)   # add state to list
            
            m.states = s_instances  # set the state link for the measure

        else:   # measure affect specific activities or pressures
            a = activity_instances[activity_id] # linked activity
            p = pressure_instances[pressure_id] # linked pressure
            activitypressure_id = activity_id + pressure_id # combined id of activity and pressure

            if activitypressure_id not in activity_instances:   # if the id is not already added
                ap = ActivityPressure(activity=a, pressure=p)   # create ActivityPressure object
                activitypressure_instances.update({activitypressure_id: ap}) # add link to dictionary
            else:
                ap = activitypressure_instances[activitypressure_id]    # access the link

            m.activity_pressure = ap    # add the link to the measure object

        # assign expected value and its uncertainty 
        expected = expecteds.loc[num] / 100.0
        uncertainty = uncertainties.loc[num+1] / 100.0

        # get expert survey probability distribution
        block_id = measures_blocks.loc[num, 'block']
        weights = expert_weights.loc[expert_weights['block'] == block_id, expert_ids]
        prob_dist = get_prob_dist(expecteds=expert_expecteds.loc[num], 
                                  lower_boundaries=expert_lower_boundaries.loc[num+2], 
                                  upper_boundaries=expert_upper_boundaries.loc[num+3], 
                                  weights=weights)

        m.expected = expected   # set expected value of the measure
        m.uncertainty = uncertainty # set uncertainty of the measure

    # Rearrange initialized core objects in DataFrame
    # and make linkages between core objects
    measure_df = pd.DataFrame.from_dict({
        'instance': measure_instances.values()
    })
 
    # ID:s are calculated so that they can be tracked
    measure_df['measure id'] = [int(x.id / 10000) * 10000 for x in measure_df['instance']]
    measure_df['activity-pressure id'] = [x.activity_pressure.id if x.activity_pressure != None else np.nan for x in measure_df['instance']]
    measure_df['activity id'] = [int(x.activity_pressure.id / 10000) * 10000 if x.activity_pressure != None else np.nan for x in measure_df['instance']]
    measure_df['pressure id'] = measure_df['activity-pressure id'] - measure_df['activity id']

    # Going through core objects and setting initial values
    # ActPress sheet values are still missing!

    for num in measure_df.index:
        m = measure_df.loc[num, 'instance']

        # measures have activity-pressure pairs or states
        if m.states:
            continue

        # setting expected value first on measure instance
        # multiplicating on activity-pressure intance based on measure instance
        # setting pressure reduction value in pressure instance  
        elif m.activity_pressure:
            ap = m.activity_pressure
            p = ap.pressure

            ap.expected = ap.expected * m.effect
            p.pressure_reduction = m.effect

        else:
            raise AttributeError("Measure instance is missing Activity-Pressure pair or State instances.")

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
    
    countries_by_basins, countries, basins = object_data['domain'].values()
    instances = {}

    for country in countries['COUNTRY']:
        country_id = countries[countries['COUNTRY'] == country].index[0]
    
        for basin in basins['Basin']:
            basin_id = basins[basins['Basin'] == basin].index[0]
       
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
        minimum = np.array(act_to_press['minimun'].iloc[i]) / 100.0
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