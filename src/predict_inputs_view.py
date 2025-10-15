import streamlit as st
import pandas as pd
import numpy as np
import datetime


optional_features = [
    'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag7', 'Sales_Lag14',
    'Sales_Lag30', 'Customers_Lag1', 'Customers_Lag7', 'SalesPerCustomer_Lag1'
]

def get_user_inputs_data_filename(relative_filepath='./data/streamlit/'):
    return relative_filepath + 'user_inputs_data.csv'

def get_user_inputs_data_df(relative_filepath='./data/streamlit/'):
    return pd.read_csv(get_user_inputs_data_filename(relative_filepath=relative_filepath))


def prepare_and_save_user_inputs_file(
    df, user_inpute_cols,
    date_cols=['Date'],
    ignore_zero_values_cols=['Promo2SinceWeek', 'Promo2SinceYear', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                            'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag7', 'Sales_Lag14',
                            'Sales_Lag30', 'Customers_Lag1', 'Customers_Lag7', 'SalesPerCustomer_Lag1'],
    relative_filepath='../data/streamlit/'):

    summary = []

    for col in user_inpute_cols:
        col_data = df[col]
        col_type = col_data.dtype
        
        # Detect boolean-like numeric
        if np.issubdtype(col_type, np.number) and set(col_data.unique()).issubset({0, 1}):
            most_freq = col_data.mode().iloc[0] if not col_data.mode().empty else 0
            summary.append({
                "Column": col,
                "Type": "boolean",
                "Min": None,
                "Max": None,
                "Unique_Values": '|'.join(['0', '1']),
                "Default_Value": most_freq
            })
        
        # Detect datetime
        elif col in date_cols:
            summary.append({
                "Column": col,
                "Type": "date",
                "Min": col_data.min(),
                "Max": col_data.max(),
                "Unique_Values": None,
                "Default_Value": None
            })
        
        # Numeric
        elif np.issubdtype(col_type, np.number):
            if (col in ignore_zero_values_cols):
                min_gt_zero = col_data[col_data > 0].min()
                min_value = min_gt_zero
                mean_val = col_data[col_data > 0].mean() if not col_data.empty else 0
            else:
                min_value = col_data.min()
                mean_val = col_data.mean() if not col_data.empty else 0

            summary.append({
                "Column": col,
                "Type": "numeric",
                "Min": min_value,
                "Max": col_data.max(),
                "Unique_Values": None,
                "Default_Value": int(mean_val)
            })
        
        # Categorical / object
        else:
            most_freq = col_data.mode().iloc[0] if not col_data.mode().empty else None
            summary.append({
                "Column": col,
                "Type": "categorical",
                "Min": None,
                "Max": None,
                "Unique_Values": '|'.join(np.array(col_data.unique(), dtype=str).tolist()),
                "Default_Value": most_freq
            })

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary)
    
    
    # Save summary_df to a CSV file
    summary_df.to_csv(get_user_inputs_data_filename(relative_filepath=relative_filepath), index=False)

    return summary_df


def format_display_default(option):
    return option

def format_display_bool(option):
    return 'Yes' if option == 1 else 'No'

def format_display_assortment(option):
    values_map = {
        'a': 'Basic',
        'b': 'Extra',
        'c': 'Extended'
    }
    if option in values_map:
        return values_map[option]
    return option

def format_display_store_type(option):
    values_map = {
        'a': 'Type A',
        'b': 'Type B',
        'c': 'Type C',
        'd': 'Type D'
    }
    if option in values_map:
        return values_map[option]
    return option

def format_display_state_holiday(option):
    values_map = {
        'a': 'Public',
        'b': 'Easter',
        'c': 'Christmas',
        '0': 'None'
    }
    if option in values_map:
        return values_map[option]
    return option

def get_format_display_func(col_name):
    if (col_name == 'StateHoliday'):
        return format_display_state_holiday
    elif (col_name == 'StoreType'):
        return format_display_store_type
    elif (col_name == 'Assortment'):
        return format_display_assortment
    else:
        return format_display_default


def get_col_label(col):
    values_map = {
        'Date': 'Date to Predict',
        'Promo': 'Is Promo Active?',
        'Promo2': 'Is Long Term Promo Active?',
        'Promo2SinceWeek': 'Long Term Promo Active Since Week',
        'Promo2SinceYear': 'Long Term Promo Active Since Year',
        'PromoInterval': 'Long Term Promo Interval',
        'CompetitionDistance': 'Distance to Competitor Store (metres)',
        'Sales_Lag1': 'Sales 1 Day ago',
        'Sales_Lag2': 'Sales 2 Days ago',
        'Sales_Lag3': 'Sales 3 Days ago',
        'Sales_Lag7': 'Sales 7 Days ago',
        'Sales_Lag14': 'Sales 14 Days ago',
        'Sales_Lag30': 'Sales 30 Days ago',
        'Customers_Lag1': 'Customers number 1 Day ago',
        'Customers_Lag7': 'Customers number 7 Days ago',
        'SalesPerCustomer_Lag1': 'Average check total 1 day ago'
    }

    if col in values_map:
        return values_map[col]
    return col

def render_daily_predict_form_features(relative_filepath='./data/streamlit/', form_key="daily_predict_form"):
    """
    Renders a Streamlit form dynamically based on metadata_df.
    
    Parameters:
        metadata_df: pd.DataFrame with columns:
            Column | Type | Min | Max | Unique_Values | Default_Value
        form_key: str, unique key for the form (useful for Streamlit state)
    
    Returns:
        dict: dictionary with user inputs {column_name: value}
    """
    metadata_df=get_user_inputs_data_df(relative_filepath=relative_filepath)
    user_inputs = {}
    optional_inputs_state = {}

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Mandatory Details')
    with col2:
        st.subheader('Optional Details')

    for i, row in metadata_df.iterrows():
        col_name = row['Column']
        col_type = str(row['Type']).lower()
        min_val = row['Min']
        max_val = row['Max']
        unique_vals = row['Unique_Values']
        default_val = row['Default_Value']

        # Alternate between columns
        target_col = col2 if col_name in optional_features else col1

        with target_col:
            if (col_name in optional_features):
                optional_inputs_state[col_name] = st.checkbox(f'I don\'t know {get_col_label(col_name)} info', value=False)
            
            # Numeric input
            if col_type == "numeric":
                if (col_name in optional_inputs_state and optional_inputs_state[col_name] == True):
                    user_inputs[col_name] = 0
                else:
                    user_inputs[col_name] = st.slider(
                        label=get_col_label(col_name),
                        min_value=float(min_val) if pd.notna(min_val) else 0,
                        max_value=float(max_val) if pd.notna(max_val) else None,
                        value=float(default_val) if pd.notna(default_val) else 0,
                        step=1.0
                    )
        
            # Categorical input OR Boolean input
            elif col_type == "categorical" and unique_vals is not None:
                unique_vals_options = unique_vals.split('|')
                default_index = unique_vals_options.index(default_val)
                format_display_func = get_format_display_func(col_name)
                user_inputs[col_name] = st.selectbox(
                    label=get_col_label(col_name),
                    options=unique_vals_options,
                    index=default_index,
                    format_func=format_display_func
                )
        
            # Boolean input
            elif col_type == "boolean" and unique_vals is not None:
                unique_vals_options = [1,0]
                default_index = unique_vals_options.index(int(default_val))
                user_inputs[col_name] = st.selectbox(
                    label=get_col_label(col_name),
                    options=[1,0],
                    index=default_index,
                    format_func=format_display_bool
                )
            
            # Date input
            elif col_type == "date":
                # Default to today
                default_date = datetime.date.today()
                user_inputs[col_name] = st.date_input(
                    label=get_col_label(col_name),
                    value=pd.to_datetime(default_date).date()
                )
    
    return user_inputs


    