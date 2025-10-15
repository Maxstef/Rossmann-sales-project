import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# -------------------------------
# Global variables
# -------------------------------
target_col = 'Sales'  # Target variable for prediction

# -------------------------------
# Train/Validation/Test Split
# -------------------------------
def split_train_val_test(df, random_state=42):
    """
    Splits the DataFrame into train, validation, and test sets based on Year and Month.
    
    Parameters:
        df (pd.DataFrame): Original dataset.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: train_df, val_df, test_df
    """
    # Train set
    train_2013_part = df[df['Year'] == 2013]
    train_2014_part = df[(df['Year'] == 2014) & (df['Month'] <= 7)]
    aug_dec_2014_part = df[(df['Year'] == 2014) & (df['Month'] > 7)]
    val_test_2015_part = df[df['Year'] == 2015]

    # Split Aug-Dec 2014 part into train and val/test
    train_2014_part2, val_test_2014_part = train_test_split(
        aug_dec_2014_part, test_size=0.15, random_state=random_state
    )

    # Split remaining parts into validation and test
    val_2014_part, test_2014_part = train_test_split(val_test_2014_part, test_size=0.5, random_state=random_state)
    val_2015_part, test_2015_part = train_test_split(val_test_2015_part, test_size=0.5, random_state=random_state)

    # Concatenate final train, val, and test sets
    train_df = pd.concat([train_2013_part, train_2014_part, train_2014_part2])
    val_df = pd.concat([val_2014_part, val_2015_part])
    test_df = pd.concat([test_2014_part, test_2015_part])

    return train_df, val_df, test_df

# -------------------------------
# Normalize outliers
# -------------------------------
def get_outliers_normalized(
    df,
    normalize_upper_setup={'Customers': 3000, 'Sales': 23000, 'SalesPerCustomer': 20}
):
    """
    Caps values of specified columns to an upper limit to handle outliers.
    
    Parameters:
        df (pd.DataFrame): Input dataset.
        normalize_upper_setup (dict): Column-wise upper limits.
        
    Returns:
        pd.DataFrame: DataFrame with capped values.
    """
    to_normalize_cols = list(normalize_upper_setup.keys())
    df_copy = df.copy()

    for col in df.columns:
        # Get original column name for lag features
        original_col = col.split('_Lag')[0] if '_Lag' in col else col

        if original_col in to_normalize_cols:
            df_copy.loc[df_copy[col] > normalize_upper_setup[original_col], col] = normalize_upper_setup[original_col]

    return df_copy

# -------------------------------
# Column selection helpers
# -------------------------------
def get_input_cols(df, to_exclude=['Sales', 'Date', 'Store', 'Open', 'CompetitionDistance', 'Customers', 'SalesPerCustomer']):
    """
    Returns a list of input feature columns excluding target and irrelevant columns.
    """
    all_cols = df.columns.to_list()
    filtered_list = [col for col in all_cols if col not in to_exclude and col != target_col]
    return filtered_list

def get_impute_cols(df, all_cols, impute_lag=True):
    """
    Returns the list of columns that should be imputed (zeros replaced by mean or other strategy).
    """
    if impute_lag:
        impute_cols = get_zero_not_expected_cols() + get_lag_features_cols(df)
    else:
        impute_cols = get_zero_not_expected_cols()
    # Keep only columns present in dataset
    filtered = [col for col in impute_cols if col in all_cols]
    return filtered

def get_zero_not_expected_cols():
    """Returns columns where zero values are not expected and need imputation."""
    return ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']

def get_lag_features_cols(df):
    """Returns columns that represent lag features."""
    numeric_cols = df.select_dtypes('number').columns.tolist()
    lag_cols = [col for col in numeric_cols if '_Lag' in col]
    return lag_cols

# -------------------------------
# Categorical encoding
# -------------------------------
def encode_categorical_cols(inputs, encoder, categorical_cols):
    """
    One-hot encodes categorical columns and appends them to the input DataFrame.
    
    Parameters:
        inputs (pd.DataFrame): Input DataFrame.
        encoder (OneHotEncoder): Fitted sklearn encoder.
        categorical_cols (list): List of categorical column names.
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    encoded = pd.DataFrame(
        encoder.transform(inputs[categorical_cols]),
        columns=encoded_cols,
        index=inputs.index
    )
    return pd.concat([inputs, encoded], axis=1)

# -------------------------------
# Preprocessor setup
# -------------------------------
def get_preprocessors_for_setup(
    train_inputs, scaler_type='normalized',
    impute=True, impute_strategy='mean', impute_lag=True
):
    """
    Sets up preprocessors (imputer, scaler, encoder) based on training data.
    
    Returns:
        tuple: numeric_cols, categorical_cols, encoded_cols, impute_cols, imputer, scaler, encoder
    """
    # Split columns into numeric and categorical
    numeric_cols = train_inputs.select_dtypes('number').columns.tolist()
    categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

    # Imputer setup
    if impute:
        imputer = SimpleImputer(strategy=impute_strategy, missing_values=0)
        impute_cols = get_impute_cols(train_inputs, numeric_cols, impute_lag=impute_lag)
        imputer.fit(train_inputs[impute_cols])
    else:
        imputer = None
        impute_cols = []

    # Scaler setup
    if scaler_type == 'normalized':
        scaler = MinMaxScaler()
        scaler.fit(train_inputs[numeric_cols])
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        scaler.fit(train_inputs[numeric_cols])
    else:
        scaler = None

    # One-hot encoder setup
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    return numeric_cols, categorical_cols, encoded_cols, impute_cols, imputer, scaler, encoder

# -------------------------------
# Prepare train/val/test data
# -------------------------------
def prepare_data(
    df, input_cols,
    target_col=target_col, scaler_type='normalized',
    impute=True, impute_strategy='mean', impute_lag=True,
    normalize_outliers=False
):
    """
    Prepares train, validation, and test data with preprocessing:
      - Imputation
      - Scaling
      - Categorical encoding
      - Optional outlier normalization
    
    Returns:
        tuple: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Split data
    train_df, val_df, test_df = split_train_val_test(df=df)

    # Normalize outliers if requested
    if normalize_outliers:
        train_df = get_outliers_normalized(df=train_df)

    # Prepare input and target
    train_inputs = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    val_inputs = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()
    test_inputs = test_df[input_cols].copy()
    y_test = test_df[target_col].copy()

    # Get preprocessors
    numeric_cols, categorical_cols, encoded_cols, impute_cols, imputer, scaler, encoder = get_preprocessors_for_setup(
        train_inputs=train_inputs,
        scaler_type=scaler_type,
        impute=impute,
        impute_strategy=impute_strategy,
        impute_lag=impute_lag
    )

    # Apply imputation
    if imputer is not None:
        train_inputs[impute_cols] = imputer.transform(train_inputs[impute_cols])
        val_inputs[impute_cols] = imputer.transform(val_inputs[impute_cols])
        test_inputs[impute_cols] = imputer.transform(test_inputs[impute_cols])

    # Apply scaling
    if scaler is not None:
        train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
        val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
        test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

    # Apply one-hot encoding
    train_inputs = encode_categorical_cols(train_inputs, encoder, categorical_cols)
    val_inputs = encode_categorical_cols(val_inputs, encoder, categorical_cols)
    test_inputs = encode_categorical_cols(test_inputs, encoder, categorical_cols)

    # Select final columns
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    X_test = test_inputs[numeric_cols + encoded_cols]

    return X_train, y_train, X_val, y_val, X_test, y_test

# -------------------------------
# Extended helper to get preprocessors directly from full dataset
# -------------------------------
def get_preprocessors_for_setup_ext(
    df, input_cols,
    target_col=target_col, scaler_type='normalized',
    impute=True, impute_strategy='mean', impute_lag=True,
    normalize_outliers=False
):
    """
    Prepares preprocessors (imputer, scaler, encoder) using full dataset.
    
    Returns:
        tuple: numeric_cols, categorical_cols, encoded_cols, impute_cols, imputer, scaler, encoder
    """
    input_cols = get_input_cols(df)
    train_df, val_df, test_df = split_train_val_test(df=df)

    if normalize_outliers:
        train_df = get_outliers_normalized(df=train_df)

    train_inputs = train_df[input_cols].copy()

    return get_preprocessors_for_setup(
        train_inputs=train_inputs,
        scaler_type=scaler_type,
        impute=impute,
        impute_strategy=impute_strategy,
        impute_lag=impute_lag
    )
