import pandas as pd
import numpy as np

def find_object_features(df):
    return list(df.dtypes[df.dtypes == 'object'].index)

def find_object_feature_values(df):
    object_features = find_object_features(df)
    return df[object_features][:1].values[0]

def find_booleans(df):
    columns = df.columns
    boolean_columns = np.array([column for column in columns if len(df[column].value_counts(dropna=True)) == 2])
    boolean_values = np.array([df[column].value_counts(dropna=True).index.to_list() for column in boolean_columns])
    columns_and_values = np.stack((boolean_columns, boolean_values[:, 0], boolean_values[:, 1])).T
    return columns_and_values


def select_booleans(df, values = []):
    boolean_columns = find_booleans(df)
    matches = np.isin(boolean_columns[:, 1], values)
    return boolean_columns[matches]



def to_booleans(df, boolean_mapping):
    boolean_values = list(boolean_mapping.keys())
    boolean_features = select_booleans(df, boolean_values)[:, 0]
    boolean_df = pd.DataFrame({})
    for feature in boolean_features:
        boolean_df[feature] = df[feature].map(boolean_mapping)
    return boolean_df[boolean_features]


def remove_punctuation(string):
    return string.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')

def matrix_new_features(df):
    bin_feats = almost_binary(df)
    new_bin_feats = np.array(['{column}_is_{top}'.format(column = column, top = remove_punctuation(top)) for column, top in bin_feats])
    return np.hstack((bin_feats[:, 0].reshape(-1, 1), bin_feats[:, 1].reshape(-1, 1), new_bin_feats.reshape(-1, 1)))

def booleans_without_top_values(df, not_values):
    potential_new_features = matrix_new_features(df)
    not_tf = ~np.isin(potential_new_features[:, 1], not_values)
    return potential_new_features[not_tf]


def almost_to_boolean(df):
    columns_to_replace = matrix_new_features(df)[:, 0]
    values_to_replace = matrix_new_features(df)[:, 1]
    new_column_names = matrix_new_features(df)[:, 2]
    to_replace_df = pd.DataFrame({})
    for column, value, new_name in zip(columns_to_replace, values_to_replace, new_column_names):
        bool_column = np.where(df[column] == value,1,0)
        to_replace_df[new_name] = bool_column
    return to_replace_df

def df_with_replaced_columns(original_df, selected_booleans_df):
    matrix_features = matrix_new_features(selected_booleans_df)

    cols_to_drop = matrix_features[:, 0]
    copied_df = original_df.copy()
    pruned_df = copied_df.drop(cols_to_drop, axis = 1)
    return pd.concat([pruned_df, selected_booleans_df], axis = 1)
