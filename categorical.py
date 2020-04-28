import pandas as pd

def find_object_features(df):
    return list(df.dtypes[df.dtypes == 'object'].index)

def find_object_feature_values(df):
    object_features = find_object_features(df)
    return df[object_features][:1].values[0]

def informative(df):
    non_informative = [column for column in df.columns if len(df[column].unique()) == 1]
    informative_columns = list(set(df.columns.to_list()) - set(non_informative))
    return df[informative_columns]

def percentage_unique(df_series):
    series_filled = df_series.dropna()
    return len(series_filled.unique())/len(series_filled)


def find_categorical(df, threshold = .5):
    categorical_df = pd.DataFrame({})
    for column in df.columns:
        if percentage_unique(df[column]) < threshold:
            categorical_df[column] = df[column]
    return categorical_df


def summarize_counts(df):
    non_empty_columns = df.dropna(axis=1,how='all').columns
    frequencies = np.array([df[column].value_counts(normalize=True).values[0] for column in non_empty_columns]).reshape(-1, 1)
    columns = non_empty_columns.to_numpy().reshape(-1, 1)
    top_values = np.array([df[column].value_counts(normalize=True).index[0] for column in non_empty_columns]).reshape(-1, 1)
    summarize = np.hstack((columns, frequencies, top_values))
    return summarize[summarize[:,1].argsort()[::-1]]



def selected_summaries(df, not_values = [], lower_bound = .1, upper_bound = 1):
    potential_cols = summarize_counts(df)
    potential_cols = potential_cols[potential_cols[:, 1] > lower_bound]
    potential_cols = potential_cols[potential_cols[:, 1] < upper_bound]
    not_tf = ~np.isin(potential_cols[:, 2], not_values)
    return potential_cols[not_tf]




def num_is_digit(array, str_index = 0):
    return np.array([value[str_index].isdigit() for value in array])




def remove_digits_from_selected(selected_matrix, col_idx, str_indices = [0, -1]):
    for idx in str_indices:
        selected_col = selected_matrix[~num_is_digit(selected_matrix[:, col_idx], idx)]
    return selected_col


def categorical_plus_values(df, threshold = 5):
    categorical_cols = find_categorical(df)
    return [column for column in categorical_cols if len(df[column].value_counts()) > threshold]


def selected_cat_values(column, threshold = .02):
    values_counted = column.value_counts(normalize=True)
    return values_counted[values_counted > threshold]


def reduce_cat_values(column, threshold = .02):
    column = column.copy()
    selected_values = selected_cat_values(column, threshold).index
    column[~column.isin(selected_values)] = 'other'
    column.astype('category')
    return column


def df_reduced_categories(df, categoricals, threshold = .01):
    new_df = pd.DataFrame()
    for category in categoricals:
        new_df[category] = reduce_cat_values(df[category], threshold)
    return new_df



def replace_df_columns(original_df, replacing_df):
    replacing_cols = replacing_df.columns
    original_df = original_df.drop(columns = replacing_cols)
    new_df = pd.concat([original_df, replacing_df], axis = 1)
    return new_df

    
