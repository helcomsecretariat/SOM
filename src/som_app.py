"""
Main SOM calculation methods.
"""

import numpy as np
import pandas as pd
import os
from som_tools import *
import api_tools
from utilities import *
import pickle
import copy

def build_input(config: dict) -> dict[str, pd.DataFrame]:
    """
    Loads input data. If loading already processed data, probability distributions need to be converted back to arrays. 

    Arguments:
        config (dict): configuration settings.
    
    Returns:
        input_data (dict): SOM input data.
    """
    # process legacy input data to be usable by the tool
    if config['use_legacy_input_data']:
        # process input data
        input_data = process_input_data(config)
    
        # load areas from layers and adjust area ids
        if config['link_mpas_to_subbasins']:
            print('Linking areas in input data...')
            input_data = api_tools.link_areas(config, input_data)
        
        # export input data to excel
        path = os.path.realpath(config['input_data_legacy']['general_input'])
        if not os.path.isfile(path): path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['input_data_legacy']['general_input'])
        path = os.path.join(os.path.dirname(path), 'input_data.xlsx')
        config['input_data']['path'] = path
        with pd.ExcelWriter(path) as writer:
            for key in input_data:
                input_data[key].to_excel(writer, sheet_name=key, index=False)
    
    # load processed input data used by the tool
    path = os.path.realpath(config['input_data']['path'])
    if not os.path.isfile(path): path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['input_data']['path'])
    input_data = pd.read_excel(io=path, sheet_name=None)
    conversion_sheet = [
        ('measure_effects', 'reduction'), 
        ('activity_contributions', 'contribution'), 
        ('pressure_contributions', 'contribution')
    ]
    for col in input_data['thresholds'].columns:
        if col not in ['state', 'area_id']:
            conversion_sheet.append(('thresholds', col))
    def str_to_arr(s):
        if type(s) is float: return s
        arr = []
        for a in [x for x in s.replace('[', '').replace(']', '').split(' ')]:
            if a != '':
                arr.append(a)
        arr = np.array(arr)
        arr = arr.astype(float)
        arr = arr / np.sum(arr)
        return arr
    for sheet in conversion_sheet:
        input_data[sheet[0]][sheet[1]] = input_data[sheet[0]][sheet[1]].apply(str_to_arr)
    
    # load areas from layers and adjust area ids (if not using legacy data)
    # if done this way, input data file is not updated
    if config['link_mpas_to_subbasins'] and not config['use_legacy_input_data']:
        print('Linking areas in input data...')
        input_data = api_tools.link_areas(config, input_data)

    # make sure areas do not go over 32 characters
    # input_data['area']['area'] = input_data['area']['area'].apply(lambda x: x if len(x) <= 32 else x[:32])
    # validate IDs
    for key in ['measure', 'activity', 'pressure', 'state', 'area']:
        input_data[key][key] = input_data[key][key].apply(sanitize_string)
    
    return input_data


def build_links(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build links by picking random samples using probability distributions.

    Arguments:
        data (dict): dict of dataframes containing all links and ids relevant to SOM calculations.

    Returns:
        data (dict): updated links and ids relevant to SOM calculations.
    """
    #
    # measure effects
    #

    # verify that there are no duplicate links
    try: assert len(data['measure_effects'][data['measure_effects'].duplicated(['measure', 'activity', 'pressure', 'state'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate measure effects in input data!', e)

    # get picks from cumulative distribution
    data['measure_effects']['reduction'] = data['measure_effects']['reduction'].apply(get_pick)
    
    #
    # activity contributions
    #

    data['activity_contributions']['contribution'] = data['activity_contributions']['contribution'].apply(get_pick)

    #
    # pressure contributions
    #

    # get picks from cumulative distribution
    data['pressure_contributions']['contribution'] = data['pressure_contributions']['contribution'].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)
    
    data['pressure_contributions'] = data['pressure_contributions'].drop_duplicates(subset=['state', 'pressure', 'area_id'], keep='first').reset_index(drop=True)

    # verify that there are no duplicate links
    try: assert len(data['pressure_contributions'][data['pressure_contributions'].duplicated(['state', 'pressure', 'area_id'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate pressure contributions in input data!', e)
    
    # make sure pressure contributions for each state / area are 100 %
    for area in data['area']['ID']:
        for state in data['state']['ID']:
            mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['state'] == state)
            relevant_contributions = data['pressure_contributions'].loc[mask, :]
            if len(relevant_contributions) > 0:
                data['pressure_contributions'].loc[mask, 'contribution'] = relevant_contributions['contribution'] / relevant_contributions['contribution'].sum()

    #
    # thresholds
    #

    threshold_cols = [x for x in data['thresholds'].columns if x not in ['state', 'area_id']]   # target thresholds

    # get picks from cumulative distribution
    for col in threshold_cols:
        data['thresholds'][col] = data['thresholds'][col].apply(lambda x: get_pick(x) if not np.any(np.isnan(x)) else np.nan)

    data['thresholds'] = data['thresholds'].drop_duplicates(subset=['state', 'area_id'], keep='first').reset_index(drop=True)

    # verify that there are no duplicate links
    try: assert len(data['thresholds'][data['thresholds'].duplicated(['state', 'area_id'])]) == 0
    except Exception as e: fail_with_message(f'Duplicate targets in input data!', e)

    return data


def build_scenario(data: dict[str, pd.DataFrame], scenario: str) -> pd.DataFrame:
    """
    Build scenario. Updates activity contributions to pressures to reflect changes in the activities.

    Arguments:
        data (dict): dict of dataframes containing all links and ids relevant to SOM calculations.
        scenario (str): name of scenario to be accessed from data.

    Returns:
        data (dict): updated links and ids relevant to SOM calculations.
    """
    act_to_press = data['activity_contributions']
    dev_scen = data['development_scenarios']

    # for each pressure, save the total contribution of activities for later normalization
    actual_sum = {}
    for pressure_id in act_to_press['pressure'].unique():
        actual_sum[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            actual_sum[pressure_id][area] = activities.loc[activities['area_id'] == area, 'contribution'].sum()
    
    # multiply activities by scenario multiplier
    def get_scenario(activity_id):
        multiplier = dev_scen.loc[dev_scen['activity'] == activity_id, scenario]
        if len(multiplier) == 0:
            return 1
        multiplier = multiplier.values[0]
        return multiplier
    act_to_press['contribution'] = act_to_press['contribution'] * act_to_press['activity'].apply(get_scenario)

    # normalize
    normalize_factor = {}
    for pressure_id in act_to_press['pressure'].unique():
        normalize_factor[pressure_id] = {}
        activities = act_to_press.loc[act_to_press['pressure'] == pressure_id, :]
        for area in activities['area_id'].unique():
            scenario_sum = activities.loc[activities['area_id'] == area, 'contribution'].sum()
            normalize_factor[pressure_id][area] = 1 + scenario_sum - actual_sum[pressure_id][area]

    def normalize(value, pressure_id, area_id):
        return value / normalize_factor[pressure_id][area_id]

    act_to_press['contribution'] = act_to_press.apply(lambda x: normalize(x['contribution'], x['pressure'], x['area_id']), axis=1)
    
    return act_to_press


def build_cases(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Builds cases.

    Arguments:
        data (dict): dict of dataframes containing all links and ids relevant to SOM calculations.

    Returns:
        data (dict): updated links and ids relevant to SOM calculations.
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


def build_changes(data: dict[str, pd.DataFrame], time_steps: int = 1, warnings: bool = False) -> dict[str, pd.DataFrame]:
    """
    Main calculation method. Simulate the reduction in activities and pressures caused by measures and 
    return the change observed in state. 

    Arguments:
        data (dict): dict of dataframes containing all links and ids relevant to SOM calculations.
        time_steps (int): NOT IMPLEMENTED. amount of time steps to simulate, i.e. applications of measures several times.
        warnings (bool): toggle for showing warnings related to calculations.

    Returns:
        data (dict): updated links and ids relevant to SOM calculations.
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

    # represents the reduction observed in the total pressure load ('ID' column)
    total_pressure_load_reductions = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(0.0)

    # same as pressure_levels, but one dataframe for each separate state, so that state specific reductions on the pressures are captured
    state_pressure_levels = {s: pd.DataFrame(data['pressure']['ID']).reindex(columns=['ID']+areas.tolist()).fillna(1.0) for s in data['state']['ID']}

    # make sure activity contributions don't exceed 100 %
    for area in areas:
        for p_i, p in pressure_levels.iterrows():
            mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['pressure'] == p['ID'])
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
            mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['state'] == s['ID'])
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
                relevant_overlaps = data['overlaps'].loc[data['overlaps']['pressure'] == p['ID'], :]    # select all overlaps affecting current pressure
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
                    for o_i, o in relevant_overlaps.loc[(relevant_overlaps['overlapped'] == m['measure']) & (relevant_overlaps['activity'] == m['activity']), :].iterrows():
                        if o['overlapping'] in relevant_measures.loc[relevant_measures['activity'] == m['activity'], 'measure'].values: # ensure the overlapping measure is also for the current activity
                            reduction = reduction * o['multiplier']
                    #
                    # contribution
                    #
                    if m['activity'] == 0:
                        contribution = 1    # if activity is 0 (= straight to pressure), contribution will be 1
                    else:
                        cont_mask = (data['activity_contributions']['activity'] == m['activity']) & (data['activity_contributions']['pressure'] == m['pressure']) & (data['activity_contributions']['area_id'] == area)
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
                        norm_mask = (data['activity_contributions']['area_id'] == area) & (data['activity_contributions']['pressure'] == p['ID'])
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
                    for o_i, o in data['overlaps'].loc[(data['overlaps']['overlapped'] == m['measure']) & (data['overlaps']['activity'] == m['activity']) & (data['overlaps']['pressure'] == m['pressure']), :].iterrows():
                        if o['overlapping'] in relevant_measures['measure'].values:
                            reduction = reduction * o['multiplier']
                    #
                    # reduce pressure
                    #
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction)

        # update state pressures from pressure levels
        for s_i, s in total_pressure_load_levels.iterrows():
            state_pressure_levels[s['ID']].loc[:, :] = pressure_levels.loc[:, :]

        # pressure contributions
        for area in areas:
            for s_i, s in total_pressure_load_levels.iterrows():    # for each state
                a_i = pressure_levels.columns.get_loc(area)     # column index of current area column
                relevant_pressures = data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['state'] == s['ID']), :]  # select contributions of pressures affecting current state in current area
                for p_i, p in relevant_pressures.iterrows():
                    #
                    # main pressure reduction
                    #
                    row_i = pressure_levels.loc[pressure_levels['ID'] == p['pressure']].index[0]
                    reduction = 1 - pressure_levels.iloc[row_i, a_i]    # reduction = 100 % - the part that is left of the pressure
                    contribution = data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['state'] == s['ID']) & (data['pressure_contributions']['pressure'] == p['pressure']), 'contribution'].values[0]
                    #
                    # subpressures
                    #
                    relevant_subpressures = data['subpressures'].loc[(data['subpressures']['state'] == s['ID']) & (data['subpressures']['state pressure'] == p['pressure']), :]     # find all rows where the current pressure acts as a state pressure for the current state
                    for sp_i, sp in relevant_subpressures.iterrows():   # for each subpressure of the current pressure
                        sp_row_i = pressure_levels.loc[pressure_levels['ID'] == sp['reduced pressure']].index[0]
                        multiplier = sp['multiplier']   # by how much does the subpressure affect the current pressure
                        red = 1 - pressure_levels.iloc[sp_row_i, a_i]    # subpressure reduction = 100 % - the part that is left of the subpressure
                        reduction = reduction + multiplier * red    # the new current pressure reduction is increased by the calculated subpressure reduction
                    try: assert reduction <= 1 + allowed_error
                    except Exception as e: fail_with_message(f'Failed on area {area}, state {s["ID"]}, pressure {p["pressure"]} with reduction {reduction}', e)
                    state_pressure_levels[s['ID']].iloc[row_i, a_i] = state_pressure_levels[s['ID']].iloc[row_i, a_i] * (1 - reduction)
                    #
                    # reduce total pressure load
                    #
                    total_pressure_load_levels.at[s_i, area] = total_pressure_load_levels.at[s_i, area] * (1 - reduction * contribution)
                    #
                    # normalize pressure contributions to reflect pressure reduction
                    #
                    if abs(1 - contribution) > allowed_error and contribution != 0:     # only normalize if there is change in contributions
                        data['pressure_contributions'].loc[(data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['state'] == s['ID']) & (data['pressure_contributions']['pressure'] == p['pressure']), 'contribution'] = contribution * (1 - reduction)   # reduce the current contribution before normalizing
                        norm_mask = (data['pressure_contributions']['area_id'] == area) & (data['pressure_contributions']['state'] == s['ID'])
                        relevant_contributions = data['pressure_contributions'].loc[norm_mask, 'contribution']
                        data['pressure_contributions'].loc[norm_mask, 'contribution'] = relevant_contributions / (1 - reduction * contribution)
                        try: assert abs(1 - data['pressure_contributions'].loc[norm_mask, 'contribution'].sum()) <= allowed_error
                        except Exception as e: fail_with_message(f'Failed on area {area}, state {s["ID"]}, pressure {p["pressure"]} with pressure contribution sum not equal to 1', e)
    
    # total reduction observed in total pressure loads
    for area in areas:
        for s_i, s in total_pressure_load_levels.iterrows():
            total_pressure_load_reductions.at[s_i, area] = 1 - total_pressure_load_levels.at[s_i, area]

    # target thresholds
    cols = [x for x in data['thresholds'].columns if x not in ['state', 'area_id']]
    thresholds = {}
    for col in cols:
        thresholds[col] = pd.DataFrame(data['state']['ID']).reindex(columns=['ID']+areas.tolist())
    for area in areas:
        a_i = total_pressure_load_levels.columns.get_loc(area)
        for s_i, s in total_pressure_load_levels.iterrows():
            row = data['thresholds'].loc[(data['thresholds']['state'] == s['ID']) & (data['thresholds']['area_id'] == area), cols]
            if len(row) == 0:
                continue
            for col in cols:
                thresholds[col].iloc[s_i, a_i] = row.loc[:, col].values[0]

    data.update({
        'pressure_levels': pressure_levels, 
        'state_pressure_levels': state_pressure_levels, 
        'total_pressure_load_levels': total_pressure_load_levels, 
        'total_pressure_load_reductions': total_pressure_load_reductions, 
        'thresholds': thresholds
    })

    return data


def set_id_columns(res: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Replaces id column values with the name of the corresponding measure/activity/pressure/state in the result dataframes.

    Arguments:
        res (dict): SOM calculation results.
        data (dict): SOM input data.

    Returns:
        res (dict): updated results.
    """
    res = copy.deepcopy(res)
    relations = {
        'Pressure': 'pressure', 
        'StatePressure': 'pressure', 
        'TPL': 'state', 
        'TPLRed': 'state'
    }
    for col in [x for x in data['thresholds'].columns if x not in ['state', 'area_id']]:
        relations['Thresholds'+str(col)] = 'state'
    def replace_ids(id, k):
        return data[k].loc[data[k]['ID'] == id, k].values[0]
    for key in relations:
        if key == 'StatePressure':
            for s in data['state']['ID']:
                for r in ['Mean', 'Error']:
                    res[key][s][r]['ID'] = res[key][s][r]['ID'].apply(lambda x: replace_ids(x, relations[key]))
                    res[key][s][r] = res[key][s][r].rename(columns={col: data['area'].loc[data['area']['ID'] == col, 'area'].values[0] for col in [c for c in res[key][s][r].columns if c != 'ID']})
        else:
            for r in ['Mean', 'Error']:
                res[key][r]['ID'] = res[key][r]['ID'].apply(lambda x: replace_ids(x, relations[key]))
                res[key][r] = res[key][r].rename(columns={col: data['area'].loc[data['area']['ID'] == col, 'area'].values[0] for col in [c for c in res[key][r].columns if c != 'ID']})
    relations = {
        'MeasureEffects': ['measure', 'activity', 'pressure', 'state'], 
        'ActivityContributions': ['activity', 'pressure', 'area_id'], 
        'PressureContributions': ['state', 'pressure', 'area_id']
    }
    conversions = {
        'activity': 'activity', 
        'pressure': 'pressure', 
        'state': 'state', 
        'area_id': 'area'
    }
    for key in relations:
        for r in ['Mean', 'Error']:
            for col in relations[key]:
                k = conversions[col] if col in conversions else col
                res[key][r][col] = res[key][r][col].apply(lambda id: data[k].loc[data[k]['ID'] == id, k].values[0] if id != 0 else '-')

    return res


def build_results(sim_res: str, input_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Process the simulated results to calculate uncertainties. 
    Uncertainty is determined as standard error of the mean.

    Arguments:
        sim_res (str): path to directory holding individual simulation run results.
        input_data (dict): SOM input data.

    Returns:
        res (dict): SOM calculation results.
    """
    files = [os.path.join(sim_res, x) for x in os.listdir(sim_res) if x.endswith('.pickle') and 'sim_res' in x]

    areas = input_data['area']['ID']
    pressures = input_data['pressure']['ID']
    states = input_data['state']['ID']

    res = {}
    
    conversion_sheet = [
        ('Pressure', 'pressure_levels', pressures), 
        ('TPL', 'total_pressure_load_levels', states), 
        ('TPLRed', 'total_pressure_load_reductions', states)
    ]
    for col in [x for x in input_data['thresholds'].columns if x not in ['state', 'area_id']]:
        conversion_sheet.append(('Thresholds'+str(col), ('thresholds', col), states))
    for key, val, ids in conversion_sheet:
        res[key] = {
            'Mean': pd.DataFrame(ids).reindex(columns=['ID']+areas.tolist()).fillna(1.0), 
            'Error': pd.DataFrame(ids).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
        }
        arr = np.empty(shape=(len(ids.tolist()), len(areas.tolist()), len(files)))
        for i in range(len(files)):
            with open(files[i], 'rb') as f:
                data = pickle.load(f)
            if type(val) == str:
                arr[:, :, i] = data[val].values[:, 1:]
            else:
                arr[:, :, i] = data[val[0]][val[1]].values[:, 1:]
        res[key]['Mean'].iloc[:, 1:] = np.mean(arr, axis=2)
        res[key]['Error'].iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])    # calculate standard error
    
    res['StatePressure'] = {
        s: {
            'Mean': pd.DataFrame(pressures).reindex(columns=['ID']+areas.tolist()).fillna(1.0), 
            'Error': pd.DataFrame(pressures).reindex(columns=['ID']+areas.tolist()).fillna(1.0)
        } for s in states
    }
    for s in res['StatePressure']:
        arr = np.empty(shape=(len(pressures.tolist()), len(areas.tolist()), len(files)))
        for i in range(len(files)):
            with open(files[i], 'rb') as f:
                data = pickle.load(f)
            arr[:, :, i] = data['state_pressure_levels'][s].values[:, 1:]
        res['StatePressure'][s]['Mean'].iloc[:, 1:] = np.mean(arr, axis=2)
        res['StatePressure'][s]['Error'].iloc[:, 1:] = np.std(arr, axis=2, ddof=1) / np.sqrt(arr.shape[2])

    for key, val, col in [
        ('MeasureEffects', 'measure_effects', 'reduction'), 
        ('ActivityContributions', 'activity_contributions', 'contribution'), 
        ('PressureContributions', 'pressure_contributions', 'contribution')
    ]:
        res[key] = {
            'Mean': pd.DataFrame(input_data[val]), 
            'Error': pd.DataFrame(input_data[val])
        }
        arr = np.empty(shape=([x for x in input_data[val].values.shape]+[len(files)]))
        for i in range(len(files)):
            with open(files[i], 'rb') as f:
                data = pickle.load(f)
            arr[:, :, i] = data[val].values
        res[key]['Mean'][col] = np.mean(arr[:, -1, :], axis=1)
        res[key]['Error'][col] = np.std(arr[:, -1, :], axis=1, ddof=1) / np.sqrt(arr.shape[2])

    return res


def export_results_to_excel(res: dict[str, pd.DataFrame], input_data: dict[str, pd.DataFrame], export_path: str):
    """
    Exports simulation results as excel file.

    Arguments:
        res (dict): SOM calculation results.
        input_data (dict): SOM input data.
        export_path (str): output path for exported results.
    """
    with pd.ExcelWriter(export_path) as writer:
        new_res = set_id_columns(res, input_data)
        for key in new_res:
            if key != 'StatePressure':
                for r in ['Mean', 'Error']:
                    new_res[key][r].to_excel(writer, sheet_name=key+r, index=False)


#EOF