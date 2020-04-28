import pandas as pd
from sklearn_pandas import FunctionTransformer, DataFrameMapper


# Add discover numbers
def contains_numbers(column):
    # matches price or percentage     
    regex_string = (r'^(?!.*www|.*-|.*\/|.*[A-Za-z]|.* ).*\d.*')
#     regex_string = (r'\$\d+.*|\d+.*\%$|^\d+.*$')
    return column.str.contains(regex_string).all()

def find_almost_num_cols(df):
    object_df = df.select_dtypes(include = 'object')
    return [col for col in object_df.columns if contains_numbers(object_df[col])]

def convert_percent(val):
    if pd.isna(val):
        return pd.to_numeric(val, errors='coerce', downcast='float')
    else:
        without_dollar = val[:-1]
        return float(without_dollar)
    
def convert_price(val):
    if pd.isnull(val):
        return pd.to_numeric(val) 
    else:
        without_commas = val.replace(',','')
        without_dollar = without_commas[1:]
        return pd.to_numeric(without_dollar)

def from_price_steps(cols):
    return [([col], [FunctionTransformer(convert_price)]) 
                         for col in cols]

def from_percent_steps(cols):
    return [ ([col], [FunctionTransformer(convert_percent)]) for col in cols]
    
def from_price_and_percent_steps(price_cols, percent_cols):
    price_steps = from_price_steps(price_cols)
    percent_steps = from_percent_steps(percent_cols)
    return price_steps + percent_steps
    
# to_number_mapper = DataFrameMapper(convert_to_nums_steps, df_out = True)

# converted_nums_df = to_number_mapper.fit_transform(X_train_df_var)
