def relevant_data(data_df):
    """
    Returns only the required columns from the data set
    :param data_df: raw pandas data frame
    :return: pandas data frame with relevant columns
    """
    data_df = data_df.drop('Application_ID', axis=1)
    return data_df


def get_x_y(data_df):
    """
    Returns X and y i.e. predictors and target variale from data set
    :param data_df: raw data frame
    :return: 2 pandas data frames
    """

    X = data_df.drop('Application_Status', axis=1)
    y = data_df.loc[:, 'Application_Status']

    return X, y


def cat2int(data_df):
    """
    Converts categorical values in to discret numeric values
    :param data_df: raw data frame
    :return: data frame with categorical converted to numerics
    """

    data_df['Dependents'] = data_df['Dependents'].map(
        lambda x: 4 if x == '3+' else int(x))

    data_df['Gender'] = data_df['Gender'].map(lambda x: 0 if x == 'No' else 1)

    data_df['Education'] = data_df['Education'].map(
        lambda x: 0 if x == 'Not Graduate' else 1)

    data_df['Married'] = data_df['Married'].map(
        lambda x: 0 if x == 'No' else 1)

    data_df['Property_Area'] = data_df['Property_Area'].map(
        lambda x: 0 if x == 'Urban' else 1 if x == 'Semiurban' else 2)

    data_df['Income'] = data_df['Income'].map(
        lambda x: 0 if x == 'low' else 1 if x == 'medium' else 2)

    data_df['Self_Employed'] = data_df['Self_Employed'].map(
        lambda x: 0 if x == 'No' else 1)

    return data_df
