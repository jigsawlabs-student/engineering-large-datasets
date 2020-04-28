from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn_pandas import DataFrameMapper

def has_na_cols(df):
    not_object_df = df.select_dtypes(exclude = 'object')
    na_cols = not_object_df.isna().any(axis = 0)
    return na_cols[na_cols == True].index


def build_null_mapper(df, cols_with_na):
    imputer_steps = [([col], SimpleImputer()) for col in cols_with_na]
    is_missing_steps = [([col], MissingIndicator(), 
                     {'alias': f'{col}_is_na'})
                    for col in cols_with_na]
    combined_steps = imputer_steps + is_missing_steps
    return DataFrameMapper(combined_steps, df_out = True)

