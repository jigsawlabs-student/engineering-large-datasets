same_cols = ['scrape_id', 'experiences_offered', 'thumbnail_url', 'medium_url',
 'xl_picture_url', 'host_acceptance_rate', 'country_code', 'country', 'has_availability',
 'jurisdiction_names', 'is_business_travel_ready']

def find_all_same(df):
    return [col for col in df.columns if len(df[col].unique()) == 1]

def drop_same_cols(same_cols, df):
    return df.drop(columns = same_cols)

