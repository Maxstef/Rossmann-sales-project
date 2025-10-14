import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

target_col = 'Sales'

####################################
###### SPLIT TRAIN TEST ##########
####################################
def split_train_val_test(df, random_state=42):
  train_2013_part = df[df['Year'] == 2013]
  train_2014_part = df[(df['Year'] == 2014) & (df['Month'] <= 7)]
  aug_dec_2014_part = df[(df['Year'] == 2014) & (df['Month'] > 7)]
  val_test_2015_part = df[df['Year'] == 2015]

  train_2014_part2, val_test_2014_part = train_test_split(aug_dec_2014_part, test_size=0.15, random_state=random_state)

  val_2014_part, test_2014_part = train_test_split(val_test_2014_part, test_size=0.5, random_state=random_state)
  val_2015_part, test_2015_part = train_test_split(val_test_2015_part, test_size=0.5, random_state=random_state)

  train_df = pd.concat([train_2013_part, train_2014_part, train_2014_part2])
  val_df = pd.concat([val_2014_part, val_2015_part])
  test_df = pd.concat([test_2014_part, test_2015_part])

  return (train_df, val_df, test_df)


####################################
###### NORMALIZE OUTLIERS ##########
####################################
def get_outliers_normalized(
    df,
    normalize_upper_setup={
      'Customers': 3000,
      'Sales': 23000,
      'SalesPerCustomer': 20
    }
  ):

  to_normilize_cols = list(normalize_upper_setup.keys())
  df_copy = df.copy()

  for col in df.columns:
    if ('_Lag' in col):
      original_col = col.split('_Lag')[0]
    else:
      original_col = col

    if (original_col in to_normilize_cols):
      df_copy.loc[df_copy[col] > normalize_upper_setup[original_col], col] = normalize_upper_setup[original_col]

  return df_copy

#########################
#### GET INPUT COLS #####
#########################
def get_input_cols(df, to_exclude=['Sales', 'Date', 'Store', 'Open', 'CompetitionDistance', 'Customers', 'SalesPerCustomer']):
  all_cols = df.columns.to_list()

  filtered_list = [col for col in all_cols if col not in to_exclude and col != target_col]
  return filtered_list

####################################
### COLUMNS TO BE IMPUTED ##########
####################################
def get_impute_cols(df, all_cols, impute_lag=True):
  if (impute_lag):
    impute_cols = get_zero_not_expected_cols() + get_lag_features_cols(df)
  else:
    impute_cols = get_zero_not_expected_cols()
  filtered = [col for col in impute_cols if col in all_cols]
  return filtered

def get_zero_not_expected_cols():
  return ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']

def get_lag_features_cols(df):
  numeric_cols = df.select_dtypes('number').columns.tolist()
  lag_cols = [col for col in numeric_cols if '_Lag' in col]
  return lag_cols

def encode_categorical_cols(inputs, encoder, categorical_cols):
  encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
  encoded = pd.DataFrame(
      encoder.transform(inputs[categorical_cols]),
      columns=encoded_cols,
      index=inputs.index
  )
  return pd.concat([inputs, encoded], axis=1)

#########################
### GET PREPROCESSORS ###
#########################
def get_preprocessors_for_setup(
    train_inputs, scaler_type='normalized',
    impute=True, impute_strategy='mean', impute_lag=True):

  # split columns to numeric and categorical
  numeric_cols = train_inputs.select_dtypes('number').columns.tolist()
  categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

  # imputer setup and fit with train data
  if (impute):
    imputer = SimpleImputer(strategy=impute_strategy, missing_values=0)
    impute_cols = get_impute_cols(train_inputs, numeric_cols, impute_lag=impute_lag)
    imputer.fit(train_inputs[impute_cols])
  else:
    imputer = None
    impute_cols = []

  # scaler setup based on type and fit with train data
  if (scaler_type == 'normalized'):
    scaler = MinMaxScaler()
    scaler.fit(train_inputs[numeric_cols])
  elif (scaler_type == 'standard'):
    scaler = StandardScaler()
    scaler.fit(train_inputs[numeric_cols])
  else:
    scaler = None

  encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  encoder.fit(train_inputs[categorical_cols])

  encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

  return (numeric_cols, categorical_cols, encoded_cols, impute_cols, imputer, scaler, encoder)


#########################
##### PREPARE DATA ######
#########################
def prepare_data(
    df, input_cols,
    target_col=target_col, scaler_type='normalized',
    impute=True, impute_strategy='mean', impute_lag=True,
    normalize_outliers=False):

  train_df, val_df, test_df = split_train_val_test(df=df)

  if (normalize_outliers):
    train_df = get_outliers_normalized(df=df)

  train_inputs = train_df[input_cols].copy()
  y_train = train_df[target_col].copy()
  val_inputs = val_df[input_cols].copy()
  y_val = val_df[target_col].copy()
  test_inputs = test_df[input_cols].copy()
  y_test = test_df[target_col].copy()

  (
    numeric_cols,
    categorical_cols,
    encoded_cols,
    impute_cols,
    imputer,
    scaler,
    encoder
  ) = get_preprocessors_for_setup(
      train_inputs=train_inputs,
      scaler_type=scaler_type,
      impute=impute,
      impute_strategy=impute_strategy,
      impute_lag=impute_lag
  )

  # impute features with 0 values where needed
  if (imputer != None):
    train_inputs[impute_cols] = imputer.transform(train_inputs[impute_cols])
    val_inputs[impute_cols] = imputer.transform(val_inputs[impute_cols])
    test_inputs[impute_cols] = imputer.transform(test_inputs[impute_cols])

  # scale values step
  if (scaler != None):
    train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

  # encode categorical values step
  train_inputs = encode_categorical_cols(train_inputs, encoder, categorical_cols)
  val_inputs = encode_categorical_cols(val_inputs, encoder, categorical_cols)
  test_inputs = encode_categorical_cols(test_inputs, encoder, categorical_cols)

  X_train = train_inputs[numeric_cols + encoded_cols]
  X_val = val_inputs[numeric_cols + encoded_cols]
  X_test = test_inputs[numeric_cols + encoded_cols]

  return (X_train, y_train, X_val, y_val, X_test, y_test)


# hepper function to prepare preprocessors
def get_preprocessors_for_setup_ext(df, input_cols,
    target_col=target_col, scaler_type='normalized',
    impute=True, impute_strategy='mean', impute_lag=True,
    normalize_outliers=False):

  input_cols=get_input_cols(df)
  train_df, val_df, test_df = split_train_val_test(df=df)

  if (normalize_outliers):
    train_df = get_outliers_normalized(df=train_df)

  train_inputs = train_df[input_cols].copy()

  return get_preprocessors_for_setup(
      train_inputs=train_inputs,
      scaler_type=scaler_type,
      impute=impute,
      impute_strategy=impute_strategy,
      impute_lag=impute_lag
  )