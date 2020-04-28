import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer, MissingIndicator

true_bools_cols = ['requires_license', 'host_has_profile_pic',
       'require_guest_profile_picture',
       'require_guest_phone_verification', 'host_is_superhost',
       'instant_bookable', 'is_location_exact', 'host_identity_verified']

true_bool_steps = [([col], [SimpleImputer(strategy = 'constant', missing_values= np.nan, fill_value = 'f'),
                  MissingIndicator(missing_values = 't')
                 ]) for col in true_bools_cols]

boolean_mapper = DataFrameMapper(true_bool_steps, df_out = True)



paired_bools = [('market', 'Berlin'),
 ('state', 'Berlin'),
 ('smart_location', 'Berlin, Germany'),
 ('city', 'Berlin')]

top_val_bool_steps = [([col], [SimpleImputer(strategy = 'constant', fill_value = 'f'),
                  MissingIndicator(missing_values = top_val)], 
          {'alias': f'{col}_is_{top_val}'}) 
         for col, top_val in paired_bools]

top_val_bool_mapper = DataFrameMapper(top_val_bool_steps, df_out = True)
cols_to_remove = [col for col, top_val in paired_bools]
