import pandas as pd
def convert_to_num(col):
    price_str = col.str.replace(',','').str[1:]
    return pd.to_numeric(price_str)

def remove_zeros_in_target(df, target_name):
    return df[~(df[target_name] < 1)]

def drop_other_price_cols(df):
    return df.drop(columns = ['weekly_price', 'monthly_price'])