from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
def selected_cat_values(column, threshold = .02):
    values_counted = column.value_counts(normalize=True)
    return values_counted[values_counted > threshold]

def reduce_cat_values(column, threshold = .02):
    column = column.copy()
    selected_values = selected_cat_values(column, threshold).index
    column[~column.isin(selected_values)] = 'other'
    column.astype('category')
    return column


cat_cols = ['street', 'bed_type', 'property_type', 
       'host_location', 'room_type', 
       'host_response_time', 'cancellation_policy',
       'neighbourhood_group_cleansed', 'host_verifications',
       'neighbourhood', 'host_neighbourhood', 'calendar_updated',
       'neighbourhood_cleansed', 'zipcode']

def reduce_cat_df(df, cat_cols):
    object_df = df.select_dtypes('object')
    remaining_cat_df = object_df[cat_cols]
    return remaining_cat_df.apply(lambda col: reduce_cat_values(col, .01))