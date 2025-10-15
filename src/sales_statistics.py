import pandas as pd
import streamlit as st

relative_filepath = './data/streamlit/'

def get_col_stats_filename(col):
    filename = 'average_sales_per_' + col + '.csv'
    return filename

def safe_read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_avg_sales_per_col_df(col):
    full_filename = relative_filepath + get_col_stats_filename(col)
    return safe_read_csv(full_filename)


def get_st_stats_column_config():
    return {
        "Month": st.column_config.TextColumn(
            "Month Name",
            help="Full month name"
        ),
        "DayOfWeek": st.column_config.TextColumn(
            "Day Name",
            help="Full weekday name"
        ),
        "Average_Sales": st.column_config.NumberColumn(
            "💰 Average Sales",
            help="Average daily sales when store was open",
            format="%.2f"
        ),
        "Total_Open_Days": st.column_config.NumberColumn(
            "🏪 Days Open",
            help="Total number of open days in the month",
            format="%d"
        ),
        "Total_Days_With_Promo": st.column_config.NumberColumn(
            "🎯 Promo Days",
            help="Total days with active promotions",
            format="%d"
        )
    }