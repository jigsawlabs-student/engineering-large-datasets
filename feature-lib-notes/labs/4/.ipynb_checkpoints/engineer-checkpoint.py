import pandas as pd
def find_booleans(boolean_values):
    return [feature for feature, value in zip(object_features, object_feature_values) if value in boolean_values]

def find_objects(df):
    return list(df.dtypes[df.dtypes == 'object'].index)

def feature_values(df, features):
    return df[features][:1].values[0]


import re
def is_date(text):
    if text:
        return bool(re.search(r'\d{4}-\d{2}-\d{2}', str(text)))

def date_features(features, feature_values):
    return [feature for feature, value in zip(features, feature_values) if is_date(value)]

def find_object_feature_values(df):
    object_features = find_object_features(df)
    return df[object_features][:2].values


def contains_numbers(column):
    # matches price or percentage
    regex_string = (r'^(?!.*www|.*-|.*\/|.*[A-Za-z]|.* ).*\d.*')
#     regex_string = (r'\$\d+.*|\d+.*\%$|^\d+.*$')
    return column.str.contains(regex_string).all()




def find_numeric_features(df):
    series_contains_number = df.apply(contains_numbers)
    return series_contains_number.index[series_contains_number.values]


def numeric_to_fix(df):
    numeric_features = find_numeric_features(df)
    return df[numeric_features].select_dtypes(exclude=['int64', 'float64'])[0:2]



def price_to_float(price):
    if type(price) == str and price[0] == '$':
        return float(price[1:].replace(',',''))

def prices_to_floats(df, price_features):
    prices_df = pd.DataFrame({})
    for feature in price_features:
        prices_df[feature] = df[feature].map(price_to_float)
    return prices_df


def merge_dfs(original_df, new_dfs):
    if not isinstance(new_dfs, list):
        new_dfs = [new_dfs]
    copied_original = original_df.copy()
    for new_df in new_dfs:
        copied_original[new_df.columns] = new_df
    return copied_original