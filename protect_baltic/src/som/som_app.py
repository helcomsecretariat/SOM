"""
Created 25102022 by Antti-Jussi Kieloaho (LUKE)
Modified 03/2024 by Camilo Hernández (HELCOM)
"""

from copy import deepcopy

import os
import numpy as np
import pandas as pd
import toml

from som.som_tools import read_survey_data, preprocess_survey_data, process_survey_data, read_core_object_descriptions
from som.som_tools import read_domain_input, read_case_input, read_linkage_descriptions, read_postprocess_data
from som.som_classes import Measure, Activity, Pressure, ActivityPressure, State, CountryBasin, Case


def process_input_data():
    """
    Reads in data and processes it in usable form.
    
    step 1a. read survey data from excel file
    step 1b. preprocess survey data
    step 1c. process survey data
    step 2. read core object descriptions
    step 3. read calculation domain descriptions
    step 4. read case input
    step 5. read linkage descriptions
    step 6. read postprocessing data

    Returns:
        survey_df (DataFrame): contains the survey data of expert panels
        object_data (dict): contains following data: 'measure', 'activity', 'pressure', 'state', 'domain', and 'postprocessing'
    """
    # read configuration file
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configuration.toml')
    with open(config_file, 'r') as f:
        config = toml.load(f)
    
    # convert sheet name string keys to integers in config
    config['measure_survey_sheets'] = {int(key): config['measure_survey_sheets'][key] for key in config['measure_survey_sheets']}

    # step 1a. read survey data from excel file
    file_name = config['input_files']['measure_effect_input']
    mteq, measure_survey_data = read_survey_data(file_name=file_name, sheet_names=config['measure_survey_sheets'])

    # step 1b. preprocess survey data
    survey_df = preprocess_survey_data(mteq=mteq, measure_survey_data=measure_survey_data)

    # step 1c. process survey data 
    survey_df = process_survey_data(survey_df=survey_df)

    # step 2. read core object descriptions
    file_name = config['input_files']['general_input']
    object_data = read_core_object_descriptions(file_name=file_name)

    # step 3. read calculation domain descriptions
    file_name = config['input_files']['general_input']
    domain_data = read_domain_input(file_name=file_name, 
                                    countries_exclude=config['domain_settings']['countries_exclude'], 
                                    basins_exclude=config['domain_settings']['basins_exclude'])

    # step 4. read case input 
    file_name = config['input_files']['general_input']
    case_data = read_case_input(file_name=file_name)

    # step 5. read linkage descriptions
    file_name = config['input_files']['general_input']
    linkage_data = read_linkage_descriptions(file_name=file_name)

    # step 6. read postprocessing data
    file_name = config['input_files']['general_input']
    postprocess_data = read_postprocess_data(file_name=file_name)

    object_data.update({
        'domain': domain_data,
        'cases': case_data,
        'linkages': linkage_data,
        'postprocessing': postprocess_data,
        })

    return survey_df, object_data


def build_core_object_model(survey_df, object_data):
    """
    Builds and initializes core object model.

    Arguments:
        survey_df (DataFrame): processed survey results
        object_data (dict): dictionary that contains following data: measure, activity, pressure, and state 
    """

    measures = survey_df['measure'].loc[(survey_df['title'] == 'expected value')]
    expecteds = survey_df['aggregated'].loc[(survey_df['title'] == 'expected value')]
    uncertainties = survey_df['aggregated'].loc[(survey_df['title'] == 'variance')]

    activities = survey_df['activity'].loc[(survey_df['title'] == 'expected value')]
    pressures = survey_df['pressure'].loc[(survey_df['title'] == 'expected value')]
    states = survey_df['state'].loc[(survey_df['title'] == 'expected value')]

    activity_ids = activities.unique()
    pressure_ids = pressures.unique()

    activity_instances = {}
    pressure_instances = {}

    for id in activity_ids:

        if id == 0:
            continue

        name = object_data['activity'][id/10000]
        a = Activity(id=id, name=name)
        activity_instances.update({id: a})

    for id in pressure_ids:

        if id == 0:
            continue
        
        name = object_data['pressure'][id]
        p = Pressure(id=id, name=name)
        pressure_instances.update({id: p})


        measure_instances = {}
        activitypressure_instances = {}

    for num in measures.index:

        # instantiate Measure
        measure_id = measures.loc[num]
        measure_name = object_data['measure'][int(measure_id/10000)]
        
        m = Measure(id=measure_id, name=measure_name)
        measure_instances.update({measure_id: m})

        # instantiate only State
        # Activity and Pressure have been instantiated above
        activity_id = activities.loc[num]
        pressure_id = pressures.loc[num]

        if int(activity_id) == 0 or pressure_id == 0:
            state_id = states.loc[num]
            
            if isinstance(state_id, list):
                s_instances = []

                for id in state_id:
                    if id == '': continue  # input files have inconsistances

                    state_name = object_data['state'][int(id)]
                    s = State(id=id, name=state_name)
                    s_instances.append(s)
            
            m.states = s_instances

        else:
            a = activity_instances[activity_id]
            p = pressure_instances[pressure_id]
            activitypressure_id = activity_id + pressure_id

            if activitypressure_id not in activity_instances:
                ap = ActivityPressure(activity=a, pressure=p)
                activitypressure_instances.update({activitypressure_id: ap})
            
            else:
                ap = activitypressure_instances[activitypressure_id]

            m.activity_pressure = ap

        # assign expected value and its uncertainty 
        expected = expecteds.loc[num] / 100.0
        uncertainty = uncertainties.loc[num+1] / 100.0

        m.expected = expected
        m.uncertainty = uncertainty

    # Rearrange initialized core objects in DataFrame
    # and make linkages between core objects
    measure_df = pd.DataFrame.from_dict({
        'instance': measure_instances.values()
    })
 
    # ID:s are calculated so that they are can be tracked
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
       
            basin_fraction = countries_by_basins.loc[(countries_by_basins['country'] == country), basin].values[0]
 
            if basin_fraction <= 0:
                continue

            basin_id = basins[basins['Basin'] == basin].index[0]

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