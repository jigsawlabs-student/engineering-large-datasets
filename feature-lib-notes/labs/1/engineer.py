def find_booleans(boolean_values):
    return [feature for feature, value in zip(object_features, object_feature_values) if value in boolean_values]

def find_objects(df):
    return list(df.dtypes[df.dtypes == 'object'].index
                
def feature_values(df, features):
                return df[features][:1].values[0]
                
import re
def is_date(text):
                if text:
                    return bool(re.search(r'\d{4}-\d{2}-\d{2}', str(text)))
                
def date_features(features, feature_values):
                return [feature for feature, value in zip(features, feature_values) if is_date(value)]