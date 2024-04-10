
import numpy as np
import pandas as pd

def read_survey_data(file_name, sheet_names) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """
    Measure survey data: Part 1

    Arguments:
        file_name (str): path of survey excel file
        sheet_names (dict): dict of survey sheet ids in file_name
    
    Return:
        mteq (DataFrame): survey data information
        measure_survey_data (dict): survey data
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


def preprocess_survey_data(mteq, measure_survey_data) -> pd.DataFrame:
    """
    Measure survey data: Part 2

    This method parses the survey sheets from webropol

    Arguments:
        mteq (DataFrame): survey data information
        measure_survey_data (dict): survey data
    
    Return:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            title: type of value ('expected value' / 'variance' / 'max effectivness')
            block: id for each set of questions within each survey, unique across all surveys
            measure: measure id of the question
            activity: activity id of the question
            pressure: pressure id of the question
            state: state id (if defined) of the question ([nan] if no state)
            1...n (=expert ids): answer value for each expert (NaN if no answer)
    """

    cols = ['survey_id', 'title', 'block', 'measure', 'activity', 'pressure', 'state']
    survey_df = pd.DataFrame(columns=cols)  # create new DataFrame

    block_number = 0    # represents the survey block

    # for every survey sheet
    for survey_id in measure_survey_data:
        
        survey_info = mteq[mteq['Survey ID'] == survey_id]  # select the rows linked to current survey

        end = 0     # represents last column index of the question set
        for row, amt in enumerate(survey_info['AMT']):
            
            end = end + (2 * amt + 1)     # end column for data
            start = end - (2 * amt)     # start column for data
            
            titles = ['expected value', 'variance'] * amt
            titles.append('max effectivness')

            measures = measure_survey_data[survey_id].iloc[0, start:end].tolist() # select current question column names as measure ids
            measures.append(np.nan) # append NaN for max effectivness (ME)

            activity_id = survey_info['Activity'].iloc[row] # select current row Activity
            activities = [activity_id] * amt * 2
            activities.append(np.nan)

            pressure_id = survey_info['Pressure'].iloc[row] # select current row Pressure
            pressures = [pressure_id] * amt * 2
            pressures.append(np.nan)

            direct_ids = survey_info['Direct_to_state'].iloc[row]   # select current row state
            if isinstance(direct_ids, str):
                directs = [[int(x) for x in direct_ids.split(';') if x != '']] * amt * 2
            elif isinstance(direct_ids, list):
                directs = [[int(x) for x in direct_ids if x != '']] * amt * 2
            elif isinstance(direct_ids, float) or isinstance(direct_ids, int):
                directs = [[direct_ids if not np.isnan(direct_ids) else np.nan]] * amt * 2
            else:
                directs = [direct_ids] * amt * 2
            directs.append(np.nan)
            
            data = measure_survey_data[survey_id].loc[1:, start:end]    # select current question answers
            data = data.transpose() # transpose so that experts are columns and measures are rows

            # add survey info to each entry in the data 
            data['survey_id'] = [survey_id] * len(data) # new column with survey_id for every row
            data['title'] = titles
            data['block'] = [block_number] * len(data)  # new column with block_number for every row
            data['measure'] = measures
            data['activity'] = activities
            data['pressure'] = pressures
            data['state'] = directs

            survey_df = pd.concat([survey_df, data], ignore_index=True, sort=False)
            block_number = block_number + 1

    return survey_df


def process_survey_data(survey_df):
    r'''
    Measure survey data: part 3

    1. Adjust expert answers by scaling factor (epsilon)

            epsilon = max_i * mu_i / gamma

        where
            max_i * mu_i = maximum expected value
            gamma = maximum effectivness

    2. Calculate mean

    3. Calculate variance

        assumed that variances are correlated

            Var(sum(X_i)) = sum(Var(X_i)) + 2 * (sum(Cov(X_x, X_y)))

        where
            i goes from 1 ... n
            x, y go as 1 <= x < y <= n

    4. New id for 'measure' and 'activity' by multiplying id by 10000

        This is done as we need to track specific measure-activity-pressure and measure-state combinations
        'pressure' and 'state' id:s are not multiplied!

    5. Scaling factor ('title' with value 'max effectivness') is removed

    Arguments:
        survey_df (DataFrame): survey data information
            survey_id: unique id for each questionnaire / survey
            title: type of value ('expected value' / 'variance' / 'max effectivness')
            block: id for each set of questions within each survey, unique across all surveys
            measure: measure id of the question
            activity: activity id of the question
            pressure: pressure id of the question
            state: state id (if defined) of the question ([nan] if no state)
            1...n (=expert ids): answer value for each expert (NaN if no answer)
    Returns:
        survey_df (DataFrame): processed survey data information
    '''
    id_multiplier = 10000

    # Step 1: adjust answers by scaling factor
    
    block_ids = survey_df.loc[:,'block'].unique()   # find unique block ids
    for b_id in block_ids:  # for each block
        block = survey_df.loc[survey_df['block'] == b_id, :]    # select all rows with current block id
        for col in block:   # for each column
            if isinstance(col, int):    # if it is an expert answer
                # select the expected values from the column
                expected_value = block.loc[block['title']=='expected value', col]
                # skip if no questions were answered
                if expected_value.isnull().all(): continue
                # find the highest value of the answers
                max_expected_value = expected_value.max()
                # find the max effectivness estimated by the expert
                max_effectivness = block.loc[block['title']=='max effectivness', col].values[0]
                # calculate scaling factor
                if np.isnan(max_effectivness):
                    # set all values to null if no max effectivness
                    survey_df.loc[survey_df['block'] == b_id, col] = np.nan
                elif max_effectivness == 0 or max_expected_value == 0:
                    # scale all expected values to 0 if max effectivness is zero or all expected values are zero
                    survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = 0
                else:
                    # get the scaling factor
                    scaling_factor = np.divide(max_expected_value, max_effectivness)
                    # divide the expected values by the new scaling factor
                    survey_df.loc[(survey_df['block'] == b_id) & (survey_df['title'] == 'expected value'), col] = np.divide(expected_value, scaling_factor)

    # Step 2: calculate mean

    # select column names corresponding to expert ids (any number between 1 and 100)
    expert_ids = survey_df.filter(regex='^(100|[1-9]?[0-9])$').columns

    # create a new 'aggregated' column for expected value rows and set their value as the mean of the expert answers per question
    survey_df.loc[survey_df['title'] == 'expected value', 'aggregated'] = survey_df.loc[survey_df['title'] == 'expected value', expert_ids].mean(axis=1)

    # Step 3: calculate aggregated variance

    # find variance values
    variances = survey_df.loc[survey_df['title'] == 'variance', expert_ids]
    # sum variances
    variance_sum = variances.sum(axis=1)
    # for each row (question)
    for i in range(len(variances)):
        print(f'i = {i}')
        # select row values that are not NaN
        a = variances.iloc[i, :].dropna()
        print(f'a = {type(a)}')
        covariances = np.cov(a).sum()
        print(covariances)
        # access 'aggregated' column 'variance' rows and set their values to calculated variance
        survey_df.loc[survey_df['title'] == 'variance', 'aggregated'] = variance_sum + 2 * covariances
    # chatgpt solution 1
    survey_df.loc[survey_df['title'] == 'variance', 'aggregated'] = 0
    for i in range(len(variances.columns)):
        survey_df.loc[survey_df['title'] == 'variance', 'aggregated'] += variances[i] + 2 * covariances.iloc[i, i+1:].sum()

    # Step 4: Update measure and activity ids

    survey_df['measure'] = survey_df['measure'] * id_multiplier
    survey_df['activity'] = survey_df['activity'] * id_multiplier

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

    # Step 5: Scaling factor removed

    survey_df = survey_df.loc[survey_df['title'] != 'max effectivness']     # remove scaling factor from survey data

    return survey_df


def read_core_object_descriptions(file_name):
    """
    Reads in model object descriptions from general input files

    - Core object descriptions
    - Model domain descriptions
    - Case descriptions
    - Linkage descriptions

    Arguments:
        file_name (str): source excel file name containing 
            'CountBas', 'Country list' and 'Basin list' sheets

    Returns:
        object_data (dict):
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
    """
    Reads in calculation domain descriptions

    Arguments:
        file_name (str): source excel file name containing 
            'CountBas', 'Country list' and 'Basin list' sheets
    
    Returns:
        domain (dict): {
            countries_by_basins (DataFrame): area fractions of basins (column) per country (row)
            countries (DataFrame): country ids
            basins (DataFrame): basin ids
        }
    """

    # country-basin links
    sheet_name = 'CountBas'
    countries_by_basins = pd.read_excel(io=file_name, sheet_name=sheet_name)
    index = countries_by_basins[countries_by_basins['country'] == 'All countries'].index # find index of 'All countries' row
    countries_by_basins.drop(index=index, inplace=True) # drop 'All countries' row (since it should be only 1:s in it)

    # countries
    sheet_name = 'Country list'
    countries = pd.read_excel(io=file_name, sheet_name=sheet_name, index_col='ID')
    index = countries[(countries.index == 0) | (countries.index == 10)].index   # find indexes of 'All countries' and 'Global (non-Baltic)' rows
    countries.drop(index=index, inplace=True)   # drop selected rows

    # basins
    sheet_name = 'Basin list'
    basins = pd.read_excel(io=file_name, sheet_name=sheet_name, index_col='ID')
    index = basins[basins.index == 0].index # find index of 'All basins' row
    basins.drop(index=index, inplace=True)  # drop 'All basins' row

    domain = {
        'countries_by_basins': countries_by_basins,
        'countries': countries,
        'basins': basins
    }

    return domain


def read_case_input(file_name, sheet_name='ActMeas'):
    """
    Reading in and processing data for cases.
    
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
    """
    Reads description of links between Measures, Activities, Pressures, and States.

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
    """
    Reads input data of activities to pressures in Baltic Sea basins. 
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