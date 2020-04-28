import pandas as pd
import numpy as np

def nas_sorted(df):
    return df.isnull().sum().sort_values(ascending = False)

def column_matches(df, column, match_value):
    return np.array([column, df[df[column] == match_value].index.to_numpy()])

def any_matches(df, match_value):
    column_idx_matches = np.array([column_matches(df, column, match_value) for column in df.columns])
    return np.array(np.array([match for match in column_idx_matches if match[1].any()]))

def view_matches(df, match_value):
    matches = any_matches(df, match_value)
    if not matches.any():
        print('NO MATCHES FOR PROVIDED VALUE')
        return pd.DataFrame()
    match_columns = matches[:, 0]
    rows = np.concatenate(matches[:, 1])
    return df[match_columns].iloc[rows]


from scipy import stats

def percentiles(column):
    z_scores = stats.zscore(column)
    # segment based on number of standard deviations away from the mean
    hist, bin_edges = np.histogram(z_scores, bins=np.arange(-3, 4, 1), density=True)
    return np.stack((hist, bin_edges[1:]))



def too_many_outliers(column, threshold = .05):
    #  expected .021 if normal distribution
    z_less_neg_two = percentiles(column)[0, 0]
    z_gt_two = percentiles(column)[0, -1]
    if z_less_neg_two > threshold or z_gt_two > threshold:
        return np.hstack((column.name, z_less_neg_two, z_gt_two))
    else:
        False



def outlier_columns(df, threshold = .05):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    outlier_columns = np.array([too_many_outliers(df[column]) for column in numeric_columns])
    return np.array([column for column in outlier_columns if column is not None])


def select_outliers(column, upper_tail = True):
    if upper_tail:
        return column[stats.zscore(column) > 2]
    else:
        return column[stats.zscore(column) < -2]


def informative(df):
    non_informative = [column for column in df.columns if len(df[column].unique()) == 1]
    informative_columns = list(set(df.columns.to_list()) - set(non_informative))
    return df[informative_columns]

def some_nans(df):
    informative_df = informative(df)
    some_nans_bools = pd.isnull(informative_df).any()
    return some_nans_bools.index[some_nans_bools]

def new_na_columns(df):
    nan_columns = some_nans(df)
    df_nans = pd.isnull(df[nan_columns])
    column_name_nas = ["{column_nan}_is_na".format(column_nan = column_nan) for column_nan in nan_columns]
    df_nans.columns = column_name_nas
    return df_nans

def new_df_with_na_cols(df):
    return pd.concat([df, new_na_columns(df)], axis = 1)
