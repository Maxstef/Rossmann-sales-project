import pandas as pd
import joblib

# -------------------------------
# Model loading
# -------------------------------
def load_model_data():
    """
    Loads the pre-trained Rossmann sales model and related preprocessing objects.

    Returns:
        dict: Contains model, input columns, numeric/categorical columns,
              scaler, imputer, and encoder.
    """
    return joblib.load('./models/rossmann_sales.joblib')


# -------------------------------
# Feature engineering functions
# -------------------------------
def calculate_date_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts date-based features from 'Date' column.

    Features created:
        - DayOfWeek (1=Monday, 7=Sunday)
        - Year, Month, Day, WeekOfYear, Quarter, DayOfYear
        - IsWeekend, IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd

    Parameters:
        df_input (pd.DataFrame): Input DataFrame with 'Date' column.

    Returns:
        pd.DataFrame: DataFrame with new date features.
    """
    df_input['Date'] = pd.to_datetime(df_input['Date'])
    df_input['DayOfWeek']      = df_input['Date'].dt.dayofweek + 1
    df_input['Year']           = df_input['Date'].dt.year
    df_input['Month']          = df_input['Date'].dt.month
    df_input['Day']            = df_input['Date'].dt.day
    df_input['WeekOfYear']     = df_input['Date'].dt.isocalendar().week
    df_input['Quarter']        = df_input['Date'].dt.quarter
    df_input['DayOfYear']      = df_input['Date'].dt.dayofyear
    df_input['IsWeekend']      = df_input['DayOfWeek'].isin([6, 7]).astype(int)
    df_input['IsMonthStart']   = df_input['Date'].dt.is_month_start.astype(int)
    df_input['IsMonthEnd']     = df_input['Date'].dt.is_month_end.astype(int)
    df_input['IsQuarterStart'] = df_input['Date'].dt.is_quarter_start.astype(int)
    df_input['IsQuarterEnd']   = df_input['Date'].dt.is_quarter_end.astype(int)
    return df_input


def calculate_competition_distance_category(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizes 'CompetitionDistance' into buckets.

    Parameters:
        df_input (pd.DataFrame): DataFrame with 'CompetitionDistance'.

    Returns:
        pd.DataFrame: DataFrame with 'CompetitionDistanceCategory' column.
    """
    bins = [-1, 1000, 2000, 10000, 50000, float('inf')]
    labels = ['Very Close', 'Close', 'Medium', 'Far', 'None']

    df_input['CompetitionDistanceCategory'] = pd.cut(
        df_input['CompetitionDistance'],
        bins=bins,
        labels=labels
    )
    return df_input


def calculate_competition_months_open(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates number of months a competitor store has been open.

    Missing or zero values in competition columns are replaced with 0.

    Parameters:
        df_input (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'CompetitionMonthsOpen'.
    """
    df_input['Date'] = pd.to_datetime(df_input['Date'])

    # Replace 0 with NA for calculation
    df_input['CompetitionOpenSinceMonth'] = df_input['CompetitionOpenSinceMonth'].replace(0, pd.NA).astype('Int64')
    df_input['CompetitionOpenSinceYear']  = df_input['CompetitionOpenSinceYear'].replace(0, pd.NA).astype('Int64')

    df_input['CompetitionMonthsOpen'] = ((df_input['Date'].dt.year - df_input['CompetitionOpenSinceYear']) * 12 +
                                         (df_input['Date'].dt.month - df_input['CompetitionOpenSinceMonth']))

    # Fill missing values with 0
    df_input['CompetitionMonthsOpen'] = df_input['CompetitionMonthsOpen'].fillna(0).astype(int)
    df_input['CompetitionOpenSinceMonth'] = df_input['CompetitionOpenSinceMonth'].fillna(0)
    df_input['CompetitionOpenSinceYear']  = df_input['CompetitionOpenSinceYear'].fillna(0)

    return df_input


def calculate_promo2_weeks(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates number of weeks a long-term Promo2 has been active.

    Missing Promo2 start info is replaced with 0.

    Parameters:
        df_input (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'Promo2Weeks'.
    """
    df_input['Date'] = pd.to_datetime(df_input['Date'])

    # Replace 0 with NA for calculation
    df_input['Promo2SinceYear'] = df_input['Promo2SinceYear'].replace(0, pd.NA).astype('Int64')
    df_input['Promo2SinceWeek'] = df_input['Promo2SinceWeek'].replace(0, pd.NA).astype('Int64')

    current_year = df_input['Date'].dt.isocalendar().year
    current_week = df_input['Date'].dt.isocalendar().week

    df_input['Promo2Weeks'] = ((current_year - df_input['Promo2SinceYear']) * 52 +
                               (current_week - df_input['Promo2SinceWeek']))

    df_input['Promo2Weeks'] = df_input['Promo2Weeks'].fillna(0).astype(int)
    df_input['Promo2SinceYear'] = df_input['Promo2SinceYear'].fillna(0)
    df_input['Promo2SinceWeek'] = df_input['Promo2SinceWeek'].fillna(0)

    return df_input


def calculate_is_promo_month(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the current month is a Promo month based on 'PromoInterval'.

    Parameters:
        df_input (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'IsPromoMonth'.
    """
    df_input['Date'] = pd.to_datetime(df_input['Date'])

    # Map month number to abbreviation
    month_abbr = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                  7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df_input['MonthStr'] = df_input['Date'].dt.month.map(month_abbr)

    # Check if current month is in PromoInterval
    df_input['IsPromoMonth'] = df_input.apply(
        lambda row: int(row['MonthStr'] in str(row['PromoInterval']).split(',')),
        axis=1
    )
    return df_input


def calculate_promo_weekend(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a feature indicating if a Promo day falls on a weekend.

    Parameters:
        df_input (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with 'PromoWeekend'.
    """
    df_input['PromoWeekend'] = (df_input['Promo'] & df_input['IsWeekend']).astype(int)
    return df_input


# -------------------------------
# Prediction pipeline
# -------------------------------
def predict_daily_from_user_input(user_input: dict) -> float:
    """
    Generates a daily sales prediction for a single user input row.

    Steps:
        - Convert input to DataFrame
        - Calculate all derived features
        - Apply imputer, scaler, and encoder
        - Make prediction using pre-trained model

    Parameters:
        user_input (dict): Dictionary with user inputs, keys match feature names.

    Returns:
        float: Predicted sales rounded to 2 decimals.
    """
    df_input = pd.DataFrame([user_input])

    # Feature engineering
    df_input = calculate_date_features(df_input)
    df_input = calculate_competition_distance_category(df_input)
    df_input = calculate_competition_months_open(df_input)
    df_input = calculate_promo2_weeks(df_input)
    df_input = calculate_is_promo_month(df_input)
    df_input = calculate_promo_weekend(df_input)

    # Load model and preprocessing objects
    model_dump = load_model_data()

    # Apply imputer for missing values
    df_input[model_dump['impute_cols']] = model_dump['imputer'].transform(df_input[model_dump['impute_cols']])

    # Scale numeric features
    df_input[model_dump['numeric_cols']] = model_dump['scaler'].transform(df_input[model_dump['numeric_cols']])

    # Encode categorical features
    df_input_encoded = pd.DataFrame(
        model_dump['encoder'].transform(df_input[model_dump['categorical_cols']]),
        columns=model_dump['encoded_cols'],
        index=df_input.index
    )
    df_input = pd.concat([df_input, df_input_encoded], axis=1)

    # Predict sales
    X_onerow = df_input[model_dump['numeric_cols'] + model_dump['encoded_cols']]
    pred_onerow = model_dump['model'].predict(X_onerow)

    return round(pred_onerow[0], 2)
