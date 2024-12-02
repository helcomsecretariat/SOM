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

from som.som_tools import process_measure_survey_data, process_pressure_survey_data
from som.som_tools import read_core_object_descriptions, read_domain_input, read_case_input, read_activity_contributions, read_overlaps, read_development_scenarios, get_pick
from som.som_classes import Measure, Activity, Pressure, ActivityPressure, State, CountryBasin, Case
from utilities import Timer, exception_traceback

def process_input_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in data and processes to usable form.

    Returns:
        measure_survey_df (DataFrame): contains the measure survey data of expert panels
        pressure_survey_df (DataFrame): contains the pressure survey data of expert panels
        object_data (dict): 
            'measure' (DataFrame):
                'ID': unique measure identifier
                'measure': name / description column
            'activity' (DataFrame):
                'ID': unique activity identifier
                'activity': name / description column
            'pressure' (DataFrame):
                'ID': unique pressure identifier
                'pressure': name / description column
            'state' (DataFrame):
                'ID': unique state identifier
                'state': name / description column
            'measure_effects' (DataFrame): measure effects on activities / pressures / states
            'pressure_contributions' (DataFrame): pressure contributions to states
            'thresholds' (DataFrame): changes in states required to meet specific target thresholds
            'domain'
            'cases'
            'activity_contributions'
            'overlaps'
            'development_scenarios'
    """
    #
    # read configuration file
    #
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
    measure_effects = process_measure_survey_data(file_name, config['measure_survey_sheets'])

    #
    # pressure survey data (combined pressure contributions and GES threshold)
    #

    file_name = config['input_files']['pressure_state_input']
    pressure_contributions, thresholds = process_pressure_survey_data(file_name, config['pressure_survey_sheets'])

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

    #
    # read case input
    #

    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['case']
    cases = read_case_input(file_name=file_name, sheet_name=sheet_name)

    #
    # read activity contribution data
    #

    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['postprocess']
    activity_contributions = read_activity_contributions(file_name=file_name, sheet_name=sheet_name)

    #
    # read overlap data
    #

    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['overlaps']
    overlaps = read_overlaps(file_name=file_name, sheet_name=sheet_name)

    #
    # read activity development scenario data
    #

    file_name = config['input_files']['general_input']
    sheet_name = config['general_input_sheets']['development_scenarios']
    development_scenarios = read_development_scenarios(file_name=file_name, sheet_name=sheet_name)

    object_data.update({
        'measure_effects': measure_effects, 
        'pressure_contributions': pressure_contributions, 
        'thresholds': thresholds, 
        'domain': domain_data, 
        'cases': cases, 
        'activity_contributions': activity_contributions, 
        'overlaps': overlaps, 
        'development_scenarios': development_scenarios
    })

    return object_data


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
        name = object_data['activity'].loc[object_data['activity']['ID']==id]['activity'].values[0]
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
        name = object_data['pressure'].loc[object_data['pressure']['ID']==id]['pressure'].values[0]
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
        name = object_data['state'].loc[object_data['state']['ID']==id]['state'].values[0]
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
        name = object_data['measure'].loc[object_data['measure']['ID']==id]['measure'].values[0]
        m = Measure(name=name, id=id)
        measures.update({id: m})

    #
    # Create activity-pressure links
    #
    activitypressure_instances = {}
    for num in msdf.index:  # for every row in the survey data

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
    measure_df['measure id'] = [x.id for x in measure_df['instance']]
    measure_df['activity-pressure id'] = [x.activity_pressure.id if x.activity_pressure != None else np.nan for x in measure_df['instance']]    
    measure_df['activity id'] = [int(x.activity_pressure.activity.id) if x.activity_pressure != None else np.nan for x in measure_df['instance']]
    measure_df['pressure id'] = [int(x.activity_pressure.pressure.id) if x.activity_pressure != None else np.nan for x in measure_df['instance']]

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

            countrybasin_id = (basin_id, country_id)
            countrybasin_name = f"{country} ({country_id}) and {basin} ({basin_id})"

            cb = CountryBasin(id=countrybasin_id, name=countrybasin_name)
            cb.basin_fraction = basin_fraction

            instances.update({countrybasin_id: cb})

    # Rearrange CountryBasin objects to DataFrame
    countrybasin_df = pd.DataFrame.from_dict({
        'instance': instances.values()
    })

    countrybasin_df['country-basin id'] = [x.id for x in countrybasin_df['instance']]
    countrybasin_df['basin id'] = [int(x.id[0]) for x in countrybasin_df['instance']]
    countrybasin_df['country id'] = [int(x.id[1]) for x in countrybasin_df['instance']]

    cases = object_data['cases']
    cases_num = cases.loc[:, 'ID'].unique()

    linkages = object_data['linkages']

    for c_num in cases_num:

        # In each case there is only one measure type
        measure_num = cases['measure'].loc[cases['ID'] == c_num].unique()[0]
        
        # choose measures from a dataframe
        measures = measure_df.loc[measure_df['measure id'] == measure_num * 10000]

        countrybasins = cases['area_id'].loc[cases['ID'] == c_num]

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


def build_links(object_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds and initializes core object model.

    Arguments:
        object_data (dict):

    Returns:
        links (DataFrame) = Measure-Activity-Pressure-State reduction table
    """
    msdf = object_data['measure_effects']
    psdf = object_data['pressure_contributions']

    # verify that there are no duplicate links
    assert len(msdf[msdf.duplicated(['measure', 'activity', 'pressure', 'state'])]) == 0

    # create a new dataframe for links
    links = pd.DataFrame(msdf)

    # get picks from cumulative distribution
    links['reduction'] = links['cumulative probability'].apply(get_pick)
    links = links.drop(columns=['cumulative probability'])

    # initialize multiplier column
    links['multiplier'] = np.ones(len(msdf))

    #
    # Overlaps (measure-measure interaction)
    #

    measure_ids = links['measure'].unique()
    overlaps = object_data['overlaps']
    for id in measure_ids:
        rows = overlaps.loc[overlaps['Overlapping'] == id, :]
        for i, row in rows.iterrows():
            overlapping_id = row['Overlapping']
            overlapped_id = row['Overlapped']
            pressure_id = row['Pressure']
            activity_id = row['Activity']
            multiplier = row['Multiplier']
            query = (links['measure'] == overlapped_id) & (links['pressure'] == pressure_id)
            if activity_id != 0:
                query = query & (links['activity'] == activity_id)
            links.loc[query, 'multiplier'] = links.loc[query, 'multiplier'] * multiplier

    return links


def build_cases(links: pd.DataFrame, object_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds cases.
    """
    
    # identify and go through each case individually
    cases = object_data['cases']
    areas = cases['area_id'].unique()

    # replace all zeros (0) in activity / pressure / state columns with full list of values
    # filter those lists to only include relevant IDs (from links)
    # finally explode to only have single IDs per row
    cols = ['activity', 'pressure', 'state']
    for col in cols:
        cases[col] = cases[col].astype(object)
    for i, row in cases.iterrows():
        maps_links = links.loc[links['measure'] == row['measure'], cols]    # select relevant measure/activity/pressure/state links
        if len(maps_links) == 0:
            cases.drop(i, inplace=True) # drop rows where measure has no effect
            continue
        for col in cols:
            cases.at[i, col] = maps_links[col].unique().tolist() if row[col] == 0 else row[col]
    for col in cols:
        cases = cases.explode(col)

    # filter out links that don't have associated reduction
    m = cases['activity'].isin(links['measure']).astype(int)
    a = cases['activity'].isin(links['activity']).astype(int)
    p = cases['pressure'].isin(links['pressure']).astype(int)
    s = cases['state'].isin(links['state']).astype(int)
    existing_links = (m & a & p & s)
    cases = cases.loc[existing_links, :]

    cases = cases.reset_index(drop=True)

    # create new dataframes for pressures and states each, one column per area_id
    # the value of each cell is the reduction in the pressure for that area
    # NOTE: the DataFrames are created on one line to avoid PerformanceWarning
    pressure_change = pd.DataFrame(object_data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    state_change = pd.DataFrame(object_data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)

    # TODO: add development over time here
    # so that it multiplies with the pressure change

    # TODO: for each area
    # for each pressure
    # for each measure from cases
    # subtract activity-pressure reduction from links (multiplied with dev and press. contribution)

    # go through each case individually
    for area in areas:
        c = cases.loc[cases['area_id'] == area, :]  # select cases for current area
        for p_i, p in pressure_change.iterrows():
            relevant_measures = c.loc[c['pressure'] == p['ID'], :]
            for m_i, m in relevant_measures.iterrows():
                mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                red = links.loc[mask, 'reduction'].values[0]
                multiplier = links.loc[mask, 'multiplier'].values[0]
                for mod in ['coverage', 'implementation']:
                    multiplier = multiplier * m[mod]
                reduction = red * multiplier
                # if activity is 0 (= straight to pressure), contribution will be 1
                if m['activity'] == 0:
                    contribution = 1
                # if activity is not in contribution list, contribution will be 0
                mask = (object_data['activity_contributions']['Activity'] == m['activity']) & (object_data['activity_contributions']['Pressure'] == m['pressure'])
                contribution = object_data['activity_contributions'].loc[mask, 'value']
                if len(contribution) == 0:
                    contribution = 0
                else:
                    contribution = contribution.values[0]
                pressure_change.at[p_i, area] = pressure_change.at[p_i, area] - reduction * contribution

    # TODO: for each area
    # for each state
    # for each measure from cases
    # subtract state reduction
    # also, subtract reduction seen in pressure multiplied with corresponding pressure contribution

    # straight to state measures
    for area in areas:
        c = cases.loc[cases['area_id'] == area, :]
        for s_i, s in state_change.iterrows():
            relevant_measures = c.loc[c['state'] == s['ID'], :]
            for m_i, m in relevant_measures.iterrows():
                mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                red = links.loc[mask, 'reduction'].values[0]
                multiplier = links.loc[mask, 'multiplier'].values[0]
                for mod in ['coverage', 'implementation']:
                    multiplier = multiplier * m[mod]
                reduction = red * multiplier
                state_change.at[s_i, area] = state_change.at[s_i, area] - reduction
    
    # pressure contributions
    pressure_contributions = object_data['pressure_contributions']
    for area in areas:
        a_i = pressure_change.columns.get_loc(area)
        for s_i, s in state_change.iterrows():
            relevant_pressures = pressure_contributions.loc[pressure_contributions['State'] == s['ID'], :]
            for p_i, p in relevant_pressures.iterrows():
                row_i = pressure_change.loc[pressure_change['ID'] == p['pressure']].index[0]
                reduction = pressure_change.iloc[row_i, a_i]
                contribution = p['average']
                state_change.at[s_i, area] = state_change.at[s_i, area] - contribution * reduction
    
    # compare state reduction to GES threshold
    thresholds = object_data['thresholds']
    cols = ['PR', '10', '25', '50']
    state_ges = {}
    for col in cols:
        state_ges[col] = pd.DataFrame(object_data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    for area in areas:
        a_i = state_change.columns.get_loc(area)
        for s_i, s in state_change.iterrows():
            row = thresholds.loc[(thresholds['State'] == s['ID']) & (thresholds['area_id'] == area), cols]
            if len(row) == 0:
                continue
            for col in cols:
                state_ges[col].iloc[s_i, a_i] = row.loc[:, col].values[0]

    return state_ges


#EOF