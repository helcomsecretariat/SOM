
import numpy as np
import pandas as pd

def read_survey_data(file_name, sheet_names):
    """ Measure survey data: Part 1

    input_files = {
        'general_input': 'data/generalInput.xlsx',
        'measure_effect_input': 'data/measureEffInput.xlsx',
        'pressure_state_input': 'data/pressStateInput.xlsx'
    }

    measure_survey_sheets = {
        0: 'MTEQ',
        1: 'MT_surv_Benthic',
        2: 'MT_surv_Birds',
        3: 'MT_surv_Fish',
        4: 'MT_surv_HZ',
        5: 'MT_surv_NIS',
        6: 'MT_surv_Noise',
        7: 'MT_surv_Mammals'
    }

    """
    
    # measure effect input Excel file

    mteq = pd.read_excel(io=file_name, sheet_name=sheet_names[0])

    # preprocess values
    mteq['Direct_to_state'] = [x.split(';') if type(x) == str else x for x in mteq['Direct_to_state']]

    measure_survey_data = {}

    for id, sheet in enumerate(sheet_names.values()):

        if id == 0:
            continue

        measure_survey_data[id] = pd.read_excel(io=file_name, sheet_name=sheet, header=None)

    return mteq, measure_survey_data


def preprocess_survey_data(mteq, measure_survey_data):
    """ Measure survey data: Part 2
    """

    cols = ['survey_id', 'title', 'block', 'measure', 'activity', 'pressure', 'state']
    survey_df = pd.DataFrame(columns=cols)

    block_number = 0

    for survey_id in measure_survey_data:
        
        survey_info = mteq[mteq.Survey_ID == survey_id]

        end = 0
        for row, amt in enumerate(survey_info['AMT']):
            
            questions = ['Q{}'.format(x+1) for x in range(amt)]

            title_names_exp = ['expected value' for x in range(amt)]
            title_names_var = ['variance' for x in range(amt)]
            
            titles = [-999 for x in range(2*amt)]
            titles[::2] = title_names_exp
            titles[1::2] = title_names_var
            titles.append('max effectivness')

            measure_ids = survey_info[questions].iloc[row, :].values

            measures = [-999 for x in range(2*amt)]
            measures[::2] = measure_ids
            measures[1::2] = measure_ids
            measures.append(np.nan)

            activity_id = survey_info['Activity'].iloc[row]

            activities = [activity_id] * 2*amt
            activities.append(np.nan)

            pressure_id = survey_info['Pressure'].iloc[row]

            pressures = [pressure_id] * 2*amt
            pressures.append(np.nan)

            direct_ids = survey_info['Direct_to_state'].iloc[row]

            if isinstance(direct_ids, str):
                directs = [x.split(';')[:-1] if not isinstance(x, float) else [x] for x in direct_ids]
            else:
                directs = [direct_ids] * 2*amt
            
            directs.append(np.nan)
            
            end = end + (2*amt + 1)
            start = end - (2*amt)
            
            data = measure_survey_data[survey_id].loc[:, start:end]

            data = data.transpose()

            data['survey_id'] = [survey_id] * len(data)
            data['title'] = titles
            data['block'] = [block_number] * len(data)
            data['measure'] = measures
            data['activity'] = activities
            data['pressure'] = pressures
            data['state'] = directs

            survey_df = pd.concat([survey_df, data], ignore_index=True, sort=False)
            block_number = block_number + 1

    return survey_df


def process_survey_data(survey_df):
    """ Measure survey data: part 3

    1. Scaling factor (\epsilon)

        $\epsilon = \frac{\max_i \mu_i}{\gamma}$, 
        where $\max_i \mu_i$ is maximum expected value and $\gamma$ is maximum effectivness.

    2. Grand mean

    3. Grand variance

        Assumed that variances are correlated
        $\mathrm{Var}(\sum_{i=1}^{n}X_i) = \sum_{i=1}^{n}\mathrm{Var}(X_i) + 2 \sum_{1 \leq i < j \leq n} \mathrm{Cov}(X_i,X_j)$

    4. new id for 'measure' and 'activity' by multiplying id by 10000

        This is done as we need to track specific measure-activity-pressure and measure-state combinations
        'pressure' and 'state' id:s are not multiplied!

    5. scaling factor ('title' with value 'max effectivness') is removed
    """

    block_ids = survey_df.loc[:,'block'].unique()

    for b_id in block_ids:
        block = survey_df.loc[survey_df['block'] == b_id, :]

        for col in block:
            if isinstance(col, int):

                expected_value = block.loc[block['title']=='expected value', col]

                if expected_value.isnull().all():
                    continue

                max_expected_value = expected_value.max()

                max_effectivness = block.loc[block['title']=='max effectivness', col].values

                scaling_factor = np.divide(max_expected_value, max_effectivness)

                survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = np.divide(expected_value, scaling_factor)


    expert_ids = survey_df.filter(regex='^(100|[1-9]?[0-9])$').columns

    survey_df.loc[survey_df['title'] == 'expected value', 'aggregated'] = survey_df.loc[survey_df['title'] == 'expected value', expert_ids].mean(axis=1)

    comp_1 = survey_df.loc[survey_df['title'] == 'variance', expert_ids].sum(axis=1)
    comp_2 = survey_df.loc[survey_df['title'] == 'variance', expert_ids]

    for i in range(len(comp_2)):
        comp_3 = comp_2.iloc[i, :].dropna()
        survey_df.loc[survey_df['title'] == 'variance', 'aggregated'] = comp_1 + 2 * np.cov(comp_3).sum()

    survey_df['measure'] = survey_df['measure'] * 10000
    survey_df['activity'] = survey_df['activity'] * 10000

    measure_ids = survey_df['measure'].unique()
    measure_ids = measure_ids[~pd.isnull(measure_ids)]

    for m_id in measure_ids:
        measures = survey_df.loc[(survey_df['measure'] == m_id) & (survey_df['title'] == 'expected value')]
        indeces = measures.index.values
        
        if len(measures) > 1:
        
            for num, id in enumerate(measures['measure']):
                new_id = id + (num + 1)

                index = indeces[num]

                survey_df.loc[index, 'measure'] = new_id
                survey_df.loc[index+1, 'measure'] = new_id

    # remove scaling factor from survey data
    survey_df = survey_df.loc[survey_df['title'] != 'max effectivness']

    return survey_df


def read_core_object_descriptions(file_name):
    """ Reads in model object descriptions from general input files

    - Core object descriptions
    - Model domain descriptions
    - Case descriptions
    - Linkage descriptions

    Arguments:
        file_name (str): source excel file name containing 
            'CountBas', 'Country list' and 'Basin list' sheets

    Returns:

    """
    general_input = pd.read_excel(io=file_name, sheet_name=None)

    def create_dict(sheet_name, id_col_name, obj_col_name):
        df = general_input[sheet_name]
        obj_dict = {}
        [obj_dict.update({id: name}) if isinstance(name, str) else obj_dict.update({id: None}) for id, name in zip(df[id_col_name], df[obj_col_name])]

        return obj_dict

    # Create dict for measures
    # convert id number muptiplying it by 1000
    measures_dict = create_dict('Measure type list', 'ID', 'Measure type')

    # Create dict for activities
    # convert id number muptiplying it by 1000
    activities_dict = create_dict('Activity list', 'ID', 'Activity')

    # Create dict for pressures
    # id number does not need converting
    pressure_dict = create_dict('Pressure list', 'ID', 'Pressure')

    # Create dict for states
    # id number does not need converting
    state_dict = create_dict('State list', 'ID', 'States')

    object_data = {
        'measure': measures_dict,
        'activity': activities_dict,
        'pressure': pressure_dict,
        'state': state_dict,
    }

    return object_data

def read_domain_input(file_name):
    """ Reads in calculation domain descriptions

    Arguments:
        file_name (str): source excel file name containing 
            'CountBas', 'Country list' and 'Basin list' sheets
    
    Returns:
    """
    
    sheet_name = 'CountBas'

    countries_by_basins = pd.read_excel(io=file_name, sheet_name=sheet_name)
    index = countries_by_basins[countries_by_basins['country'] == 'All countries'].index
    countries_by_basins.drop(index=index, inplace=True)

    sheet_name = 'Country list'

    countries = pd.read_excel(io=file_name, sheet_name=sheet_name, index_col='ID')
    index = countries[(countries.index == 0) | (countries.index == 10)].index
    countries.drop(index=index, inplace=True)

    sheet_name = 'Basin list'

    basins = pd.read_excel(io=file_name, sheet_name=sheet_name, index_col='ID')
    index = basins[basins.index == 0].index
    basins.drop(index=index, inplace=True)

    domain = {
        'countries_by_basins': countries_by_basins,
        'countries': countries,
        'basins': basins
    }

    return domain

def read_case_input(file_name, sheet_name='ActMeas'):
    """ Reading in and processing data for cases.
    
    Each row represents one case. 
    
    In columns of 'ActMeas' sheet ('in_Activities', 'in_Pressure' and 'In_State_components') 0 == 'all relevant'.
    Relevant activities, pressures and state can be found in measure-wise from 'MT_to_A_to_S' sheets (linkages)
    
    - multiply MT_ID id by 10000 to get right measure_id
    - multiply In_Activities ids by 10000 to get right activity_id
    - multiply B_ID by 1000 to get right basin_id

    Arguments:
        file_name (str): name of source excel file name containing 'ActMeas' sheet
        sheet_name (str): name of sheet in source excel ('ActMeas')
    """

    cases = pd.read_excel(io=file_name, sheet_name=sheet_name)
    
    
    cases['In_Activities'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['In_Activities']]
    cases = cases.explode('In_Activities')

    cases['In_Pressure'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['In_Pressure']]
    cases = cases.explode('In_Pressure')

    cases['B_ID'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['B_ID']]
    cases = cases.explode('B_ID')

    cases['C_ID'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['C_ID']]
    cases = cases.explode('C_ID')

    cases = cases.astype({
        'In_Activities': 'int',
        'In_Pressure': 'int',
        'B_ID': 'int',
        'C_ID': 'int'
        })

    cases['countrybasin_id'] = cases['B_ID'] * 1000 + cases['C_ID']

    # In_State_components is in input data just for book keeping
    cases['In_State_components'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in cases['In_State_components']]

    return cases


def read_linkage_descriptions(file_name, sheet_name='MT_to_A_to_S'):
    """ Reads description of links between Measures, Activities, Pressures, and States.

     Arguments:
        file_name (str): name of source excel file name containing 'ActMeas' sheet
        sheet_name (str): name of sheet in source excel ('ActMeas')

    Returns:
        linkages (DataFrame): dataframe containing mappings between measures to actitivities to pressures to states
    """
    
    linkages = pd.read_excel(io=file_name, sheet_name=sheet_name)
    linkages['MT'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['MT']]

    linkages['Activities'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['Activities']]
    linkages = linkages.explode('Activities')
    linkages['Activities'].notna().astype('int')

    linkages['Pressure'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['Pressure']]
    linkages = linkages.explode('Pressure')
    linkages['Pressure'].notna().astype('int')

    linkages['State (if needed)'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in linkages['State (if needed)']]

    return linkages

def read_postprocess_data(file_name, sheet_name='ActPres'):
    """ Reads input data of activities to pressures in Baltic Sea basins. 
    """
    act_to_press = pd.read_excel(file_name, sheet_name=sheet_name)

    act_to_press['expected'] = act_to_press.filter(regex='Ml[1-6]').values.tolist()
    act_to_press['minimun'] = act_to_press.filter(regex='Min[1-6]').values.tolist()
    act_to_press['maximum'] = act_to_press.filter(regex='Max[1-6]').values.tolist()

    act_to_press.drop(act_to_press.filter(regex='Ml[1-6]').columns, axis=1, inplace=True)
    act_to_press.drop(act_to_press.filter(regex='Min[1-6]').columns, axis=1, inplace=True)
    act_to_press.drop(act_to_press.filter(regex='Max[1-6]').columns, axis=1, inplace=True)

    act_to_press['Basins'] = [list(filter(None, x.split(';'))) if type(x) == str else x for x in act_to_press['Basins']]
    act_to_press = act_to_press.explode('Basins')

    return act_to_press


#EOF