"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

import numpy as np
import pandas as pd
import os
from som_tools import *
from utilities import *

def process_input_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads in data and processes to usable form.

    Arguments:
        config (dict): dictionary loaded from configuration file

    Returns:
        measure_survey_df (DataFrame): contains the measure survey data of expert panels
        pressure_survey_df (DataFrame): contains the pressure survey data of expert panels
        data (dict): container for general data dataframes
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
            'cases'
            'activity_contributions'
            'overlaps'
            'development_scenarios'
            'subpressures'
    """
    #
    # measure survey data
    #

    file_name = os.path.realpath(config['input_data']['measure_effect_input'])
    if not os.path.isfile(file_name): file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['input_data']['measure_effect_input'])
    measure_effects = process_measure_survey_data(file_name)

    #
    # pressure survey data (combined pressure contributions and GES threshold)
    #

    file_name = os.path.realpath(config['input_data']['pressure_state_input'])
    if not os.path.isfile(file_name): file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['input_data']['pressure_state_input'])
    pressure_contributions, thresholds = process_pressure_survey_data(file_name)

    #
    # measure / pressure / activity / state links
    #

    # read core object descriptions
    # i.e. ids for measures, activities, pressures and states
    file_name = os.path.realpath(config['input_data']['general_input'])
    if not os.path.isfile(file_name): file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['input_data']['general_input'])
    id_sheets = config['input_data']['general_input_sheets']['ID']
    data = read_ids(file_name=file_name, id_sheets=id_sheets)

    #
    # read case input
    #

    sheet_name = config['input_data']['general_input_sheets']['case']
    cases = read_cases(file_name=file_name, sheet_name=sheet_name)

    #
    # read activity contribution data
    #

    sheet_name = config['input_data']['general_input_sheets']['postprocess']
    activity_contributions = read_activity_contributions(file_name=file_name, sheet_name=sheet_name)

    #
    # read overlap data
    #

    sheet_name = config['input_data']['general_input_sheets']['overlaps']
    overlaps = read_overlaps(file_name=file_name, sheet_name=sheet_name)

    #
    # read activity development scenario data
    #

    sheet_name = config['input_data']['general_input_sheets']['development_scenarios']
    development_scenarios = read_development_scenarios(file_name=file_name, sheet_name=sheet_name)

    #
    # read subpressures links
    #

    sheet_name = config['input_data']['general_input_sheets']['subpressures']
    subpressures = read_subpressures(file_name=file_name, sheet_name=sheet_name)

    data.update({
        'measure_effects': measure_effects, 
        'pressure_contributions': pressure_contributions, 
        'thresholds': thresholds, 
        'cases': cases, 
        'activity_contributions': activity_contributions, 
        'overlaps': overlaps, 
        'development_scenarios': development_scenarios, 
        'subpressures': subpressures
    })

    return data


def build_links(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build links by picking random samples using probability distributions.
    """
    #
    # measure effects
    #

    # verify that there are no duplicate links
    try: assert len(data['measure_effects'][data['measure_effects'].duplicated(['measure', 'activity', 'pressure', 'state'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate measure effects in input data!', e)

    # create a new dataframe for links
    links = pd.DataFrame(data['measure_effects'])

    # get picks from cumulative distribution
    links['reduction'] = links['probability'].apply(get_pick)
    links = links.drop(columns=['probability'])

    data['measure_effects'] = links
    
    #
    # activity contributions
    #

    data['activity_contributions']['contribution'] = data['activity_contributions']['contribution'].apply(get_pick)

    #
    # pressure contributions
    #

    # get picks from cumulative distribution
    data['pressure_contributions']['contribution'] = data['pressure_contributions']['contribution'].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)
    
    # split areas into separate rows
    data['pressure_contributions'] = data['pressure_contributions'].explode('area_id')
    data['pressure_contributions'] = data['pressure_contributions'].reset_index(drop=True)

    data['pressure_contributions'] = data['pressure_contributions'].drop_duplicates(subset=['State', 'pressure', 'area_id'], keep='first').reset_index(drop=True)

    # verify that there are no duplicate links
    try: assert len(data['pressure_contributions'][data['pressure_contributions'].duplicated(['State', 'pressure', 'area_id'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate pressure contributions in input data!', e)
    
    # make sure pressure contributions for each state / area are 100 %
    for area in data['area']['ID']:
        for state in data['state']['ID']:
            mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == state)
            relevant_contributions = data['pressure_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                data['pressure_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / relevant_contributions['contribution'].sum()

    #
    # thresholds
    #

    threshold_cols = ['PR', '10', '25', '50']   # target thresholds (PR=GES)

    # get picks from cumulative distribution
    for col in threshold_cols:
        data['thresholds'][col] = data['thresholds'][col].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)

    # split areas into separate rows
    data['thresholds'] = data['thresholds'].explode('area_id')
    data['thresholds'] = data['thresholds'].reset_index(drop=True)

    data['thresholds'] = data['thresholds'].drop_duplicates(subset=['State', 'area_id'], keep='first').reset_index(drop=True)

    # verify that there are no duplicate links
    try: assert len(data['thresholds'][data['thresholds'].duplicated(['State', 'area_id'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate GES targets in input data!', e)

    return data


def build_scenario(data: dict[str, pd.DataFrame], scenario: str) -> pd.DataFrame:
    """
    Build scenario
    """
    act_to_press = data['activity_contributions']
    dev_scen = data['development_scenarios']

    # for each pressure, save the total contribution of activities for later normalization
    actual_sum = {}
    for pressure_id in act_to_press['Pressure'].unique():
        actual_sum[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['Pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            actual_sum[pressure_id][area] = activities.loc[activities['area_id'] == area, 'contribution'].sum()
    
    # multiply activities by scenario multiplier
    def get_scenario(activity_id):
        multiplier = dev_scen.loc[dev_scen['Activity'] == activity_id, scenario]
        if len(multiplier) == 0:
            return 1
        multiplier = multiplier.values[0]
        return multiplier
    act_to_press['contribution'] = act_to_press['contribution'] * act_to_press['Activity'].apply(get_scenario)

    # normalize
    normalize_factor = {}
    for pressure_id in act_to_press['Pressure'].unique():
        normalize_factor[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['Pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            scenario_sum = activities.loc[activities['area_id'] == area, 'contribution'].sum()
            normalize_factor[pressure_id][area] = 1 + scenario_sum - actual_sum[pressure_id][area]

    def normalize(value, pressure_id, area_id):
        return value * normalize_factor[pressure_id][area_id]

    act_to_press['contribution'] = act_to_press.apply(lambda x: normalize(x['contribution'], x['Pressure'], x['area_id']), axis=1)
    
    return act_to_press


def build_cases(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds cases.
    """
    cases = data['cases']
    links = data['measure_effects']
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
    
    cases = cases.reset_index(drop=True)

    # filter out links that don't have associated reduction
    m = cases['measure'].isin(links['measure'])
    a = cases['activity'].isin(links['activity'])
    p = cases['pressure'].isin(links['pressure'])
    s = cases['state'].isin(links['state'])
    existing_links = (m & a & p & s)
    cases = cases.loc[existing_links, :]

    cases = cases.reset_index(drop=True)

    # remove duplicate measures in areas, measure with highest coverage and implementation is chosen
    cases = cases.sort_values(by=['coverage', 'implementation'], ascending=[False, False])
    cases = cases.drop_duplicates(subset=['measure', 'activity', 'pressure', 'state', 'area_id'], keep='first')
    cases = cases.reset_index(drop=True)

    data['cases'] = cases

    return data


def build_changes(data: dict[str, pd.DataFrame], time_steps: int = 1, warnings = False) -> pd.DataFrame:
    """
    Simulate the reduction in activities and pressures caused by measures and 
    return the change observed in state. 
    """
    # this variable is used in assertions where float number error might affect comparisons
    allowed_error = 0.00001     

    cases = data['cases']
    links = data['measure_effects']
    areas = data['area']['ID']

    # create dataframes to store changes in pressure and state, one column per area_id
    # NOTE: the DataFrames are created on one line to avoid PerformanceWarning

    # represents the amount of the pressure ('ID' column) that is left
    # 1 = unchanged pressure, 0 = no pressure left
    pressure_levels = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    # represents the amount of the total pressure load that is left affecting the given state ('ID' column)
    # 1 = unchanged pressure load, 0 = no pressure load left affecting the state
    total_pressure_load_levels = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)

    # represents the reduction observed in the pressure ('ID' column)
    pressure_reductions = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)
    # represents the reduction observed in the total pressure load ('ID' column)
    total_pressure_load_reductions = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)

    # make sure activity contributions don't exceed 100 %
    for area in areas:
        for p_i, p in pressure_levels.iterrows():
            mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['Pressure'] == p['ID'])
            relevant_contributions = data['activity_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                contribution_sum = relevant_contributions['contribution'].sum()
                if contribution_sum > 1:
                    data['activity_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / contribution_sum
            try: assert data['activity_contributions'].loc[mask, 'contribution'].sum() <= 1 + allowed_error
            except Exception as e: fail_with_message(f'Failed to verify that activity contributions do not exceed 100 % for area {area}, pressure {p["ID"]} with contribution sum {data['activity_contributions'].loc[mask, 'contribution'].sum()}', e)

    # make sure pressure contributions don't exceed 100 %
    for area in areas:
        for s_i, s in total_pressure_load_levels.iterrows():
            mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID'])
            relevant_contributions = data['pressure_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                contribution_sum = relevant_contributions['contribution'].sum()
                if contribution_sum > 1:
                    data['pressure_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / contribution_sum
            try: assert data['pressure_contributions'].loc[mask, 'contribution'].sum() <= 1 + allowed_error
            except Exception as e: fail_with_message(f'Failed to verify that pressure contributions do not exceed 100 % for area {area}, state {s["ID"]} with contribution sum {data['pressure_contributions'].loc[mask, 'contribution'].sum()}', e)

    #
    # simulation loop
    #

    for time_step in range(time_steps):

        #
        # pressure reductions
        #

        # activity contributions
        for area in areas:
            c = cases.loc[cases['area_id'] == area, :]  # select cases for current area
            for p_i, p in pressure_levels.iterrows():
                relevant_measures = c.loc[c['pressure'] == p['ID'], :]  # select all measures affecting the current pressure in the current area
                relevant_overlaps = data['overlaps'].loc[data['overlaps']['Pressure'] == p['ID'], :]    # select all overlaps affecting current pressure
                for m_i, m in relevant_measures.iterrows():
                    #
                    # get measure effect (= reduction), and apply modifiers
                    #
                    mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                    row = links.loc[mask, :]    # find the reduction of the current measure implementation
                    if len(row) == 0:
                        if warnings: print(f'WARNING! Effect of measure {m["measure"]} on activity {m["activity"]} and pressure {m["pressure"]} not known! Measure {m["measure"]} will be skipped in area {area}.')
                        continue    # skip measure if data on the effect is not known
                    try: assert len(row) == 1
                    except Exception as e: fail_with_message(f'ERROR! Multiple instances of measure {m["measure"]} effect on activity {m["activity"]} and pressure {m["pressure"]} given in input data!', e)
                    reduction = row['reduction'].values[0]
                    for mod in ['coverage', 'implementation']:
                        reduction = reduction * m[mod]
                    #
                    # overlaps (measure-measure interaction)
                    #
                    for o_i, o in relevant_overlaps.loc[(relevant_overlaps['Overlapped'] == m['measure']) & (relevant_overlaps['Activity'] == m['activity']), :].iterrows():
                        if o['Overlapping'] in relevant_measures.loc[relevant_measures['activity'] == m['activity'], 'measure'].values: # ensure the overlapping measure is also for the current activity
                            reduction = reduction * o['Multiplier']
                    #
                    # contribution
                    #
                    if m['activity'] == 0:
                        contribution = 1    # if activity is 0 (= straight to pressure), contribution will be 1
                    else:
                        cont_mask = (data['activity_contributions']['Activity'] == m['activity']) & (data['activity_contributions']['Pressure'] == m['pressure']) & (data['activity_contributions']['area_id'] == area)
                        contribution = data['activity_contributions'].loc[cont_mask, 'contribution']
                        if len(contribution) == 0:
                            if warnings: print(f'WARNING! Contribution of activity {m["activity"]} to pressure {m["pressure"]} not known! Measure {m["measure"]} will be skipped in area {area}.')
                            continue    # skip measure if activity is not in contribution list
                        else:
                            try: assert len(contribution) == 1
                            except Exception as e: fail_with_message(f'ERROR! Multiple instances of activity {m["activity"]} contribution on pressure {m["pressure"]} given in input data!', e)
                            contribution = contribution.values[0]
                    #
                    # reduce pressure
                    #
                    pressure_levels.at[p_i, area] = pressure_levels.at[p_i, area] * (1 - reduction * contribution)
                    if pressure_levels.at[p_i, area] < 0:
                        print(f'area {area}, pressure {p["ID"]} => level = {pressure_levels.at[p_i, area]}, red = {reduction}, cont = {contribution}')
                    #
                    # normalize activity contributions to reflect pressure reduction
                    #
                    if abs(1 - contribution) > allowed_error and contribution != 0:     # only normalize if there is change in contributions
                        data['activity_contributions'].loc[cont_mask, 'contribution'] = contribution * (1 - reduction)   # reduce the current contribution before normalizing
                        norm_mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['Pressure'] == p['ID'])
                        relevant_contributions = data['activity_contributions'].loc[norm_mask, 'contribution']
                        data['activity_contributions'].loc[norm_mask, 'contribution'] = relevant_contributions / (1 - reduction * contribution)

        #
        # total pressure load reductions
        #

        # straight to state measures
        for area in areas:
            c = cases.loc[cases['area_id'] == area, :]  # select cases for current area
            for s_i, s in total_pressure_load_levels.iterrows():
                relevant_measures = c.loc[c['state'] == s['ID'], :] # select all measures affecting current state in current the area
                for m_i, m in relevant_measures.iterrows():
                    #
                    # get measure effect (= reduction), and apply modifiers
                    #
                    mask = (links['measure'] == m['measure']) & (links['activity'] == m['activity']) & (links['pressure'] == m['pressure']) & (links['state'] == m['state'])
                    row = links.loc[mask, :]
                    if len(row) == 0:
                        continue
                    reduction = row['reduction'].values[0]
                    for mod in ['coverage', 'implementation']:
                        reduction = reduction * m[mod]
                    #
                    # overlaps (measure-measure interaction)
                    #
                    for o_i, o in data['overlaps'].loc[(data['overlaps']['Overlapped'] == m['measure']) & (data['overlaps']['Activity'] == m['activity']) & (data['overlaps']['Pressure'] == m['pressure']), :].iterrows():
                        if o['Overlapping'] in relevant_measures['measure'].values:
                            reduction = reduction * o['Multiplier']
                    #
                    # reduce pressure
                    #
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction)

        # pressure contributions
        for area in areas:
            a_i = pressure_levels.columns.get_loc(area)
            for s_i, s in total_pressure_load_levels.iterrows():
                relevant_pressures = data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID']), :]  # select contributions of pressures affecting current state in current area
                for p_i, p in relevant_pressures.iterrows():
                    #
                    # main pressure reduction
                    #
                    row_i = pressure_levels.loc[pressure_levels['ID'] == p['pressure']].index[0]
                    reduction = 1 - pressure_levels.iloc[row_i, a_i]    # reduction = 100 % - the part that is left of the pressure
                    contribution = data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID']) & (data['pressure_contributions']['pressure'] == p['pressure']), 'contribution'].values[0]
                    #
                    # subpressures
                    #
                    relevant_subpressures = data['subpressures'].loc[(data['subpressures']['State'] == s['ID']) & (data['subpressures']['State pressure'] == p['pressure']), :]
                    for sp_i, sp in relevant_subpressures.iterrows():
                        sp_row_i = pressure_levels.loc[pressure_levels['ID'] == sp['Reduced pressure']].index[0]
                        multiplier = sp['Multiplier']
                        red = 1 - pressure_levels.iloc[sp_row_i, a_i]    # reduction = 100 % - the part that is left of the pressure
                        reduction = reduction + multiplier * red
                    try: assert reduction <= 1 + allowed_error
                    except Exception as e: fail_with_message(f'Failed on area {area}, state {s["ID"]}, pressure {p["pressure"]} with reduction {reduction}', e)
                    #
                    # reduce total pressure load
                    #
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction * contribution)
                    #
                    # normalize pressure contributions to reflect pressure reduction
                    #
                    if abs(1 - contribution) > allowed_error and contribution != 0:     # only normalize if there is change in contributions
                        data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID']) & (data['pressure_contributions']['pressure'] == p['pressure']), 'contribution'] = contribution * (1 - reduction)   # reduce the current contribution before normalizing
                        norm_mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['State'] == s['ID'])
                        relevant_contributions = data['pressure_contributions'].loc[norm_mask, 'contribution']
                        data['pressure_contributions'].loc[norm_mask, 'contribution'] = relevant_contributions / (1 - reduction * contribution)
                        try: assert abs(1 - data['pressure_contributions'].loc[norm_mask, 'contribution'].sum()) <= allowed_error
                        except Exception as e: fail_with_message(f'Failed on area {area}, state {s["ID"]}, pressure {p["pressure"]} with pressure contribution sum not equal to 1', e)
    
    # total reduction observed in total pressure loads
    for area in areas:
        for s_i, s in total_pressure_load_levels.iterrows():
            total_pressure_load_reductions.at[s_i, area] = 1 - total_pressure_load_levels.at[s_i, area]

    # GES thresholds
    cols = ['PR', '10', '25', '50']
    thresholds = {}
    for col in cols:
        thresholds[col] = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    for area in areas:
        a_i = total_pressure_load_levels.columns.get_loc(area)
        for s_i, s in total_pressure_load_levels.iterrows():
            row = data['thresholds'].loc[(data['thresholds']['State'] == s['ID']) & (data['thresholds']['area_id'] == area), cols]
            if len(row) == 0:
                continue
            for col in cols:
                thresholds[col].iloc[s_i, a_i] = row.loc[:, col].values[0]
    
    data.update({
        'pressure_levels': pressure_levels, 
        'total_pressure_load_levels': total_pressure_load_levels, 
        'total_pressure_load_reductions': total_pressure_load_reductions, 
        'thresholds': thresholds
    })

    return data


def build_results(sim_res: str, data: dict[str, pd.DataFrame]):
    """
    Process the simulated results to calculate uncertainties.

    Uncertainty is determined as standard error of the mean.
    """
    files = [os.path.join(sim_res, x) for x in os.listdir(sim_res) if x.endswith('.xlsx') and 'sim_res' in x]

    areas = data['area']['ID']
    pressures = data['pressure']['ID']
    states = data['state']['ID']

    #
    # pressure levels
    #
    pressure_levels_average = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    pressure_levels_error = pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    arr = np.empty(shape=(len(pressures.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='PressureLevels')
        arr[:, :, i] = df.values[:, 1:]
    pressure_levels_average.iloc[:, 1:] = np.mean(arr, axis=2)
    pressure_levels_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # total pressure load levels
    #
    total_pressure_load_levels_average = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    total_pressure_load_levels_error = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
    arr = np.empty(shape=(len(states.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='TPLLevels')
        arr[:, :, i] = df.values[:, 1:]
    total_pressure_load_levels_average.iloc[:, 1:] = np.mean(arr, axis=2)
    total_pressure_load_levels_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # thresholds
    #
    thresholds_average = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    thresholds_error = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    arr = np.empty(shape=(len(states.tolist()), len(areas.tolist()), len(files)))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='RequiredReductionsForGES')
        arr[:, :, i] = df.values[:, 1:]
    thresholds_average.iloc[:, 1:] = np.mean(arr, axis=2)
    thresholds_error.iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    #
    # measure effects
    #
    measure_effects_mean = pd.DataFrame(data['measure_effects']).drop(columns=['probability'])
    measure_effects_error = pd.DataFrame(data['measure_effects']).drop(columns=['probability'])
    arr = np.empty(shape=([x for x in data['measure_effects'].values.shape]+[len(files)]))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='MeasureEffects')
        arr[:, :, i] = df.values
    measure_effects_mean['reduction'] = np.mean(arr[:, -1, :], axis=1)
    measure_effects_error['reduction'] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])
    #
    # activity contributions
    #
    activity_contributions_mean = pd.DataFrame(data['activity_contributions'])
    activity_contributions_error = pd.DataFrame(data['activity_contributions'])
    arr = np.empty(shape=([x for x in data['activity_contributions'].values.shape]+[len(files)]))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='ActivityContributions')
        arr[:, :, i] = df.values
    activity_contributions_mean['contribution'] = np.mean(arr[:, -1, :], axis=1)
    activity_contributions_error['contribution'] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])
    #
    # pressure contributions
    #
    pressure_contributions_mean = pd.DataFrame(data['pressure_contributions'])
    pressure_contributions_error = pd.DataFrame(data['pressure_contributions'])
    arr = np.empty(shape=([x for x in data['pressure_contributions'].values.shape]+[len(files)]))
    for i in range(len(files)):
        df = pd.read_excel(io=files[i], sheet_name='PressureContributions')
        arr[:, :, i] = df.values
    pressure_contributions_mean['contribution'] = np.mean(arr[:, -1, :], axis=1)
    pressure_contributions_error['contribution'] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])

    # create new dict of dataframes
    res = {
        'PressureMean': pressure_levels_average, 
        'PressureError': pressure_levels_error, 
        'TPLMean': total_pressure_load_levels_average, 
        'TPLError': total_pressure_load_levels_error, 
        'ThresholdsMean': thresholds_average, 
        'ThresholdsError': thresholds_error, 
        'MeasureEffectsMean': measure_effects_mean, 
        'MeasureEffectsError': measure_effects_error, 
        'ActivityContributionsMean': activity_contributions_mean, 
        'ActivityContributionsError': activity_contributions_error, 
        'PressureContributionsMean': pressure_contributions_mean, 
        'PressureContributionsError': pressure_contributions_error
    }

    return res

#EOF