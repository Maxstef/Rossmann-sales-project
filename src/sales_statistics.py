import pandas as pd
import streamlit as st

# -------------------------------
# Global variables
# -------------------------------
relative_filepath = './data/streamlit/'  # Base path for CSV files

# -------------------------------
# File handling
# -------------------------------
def get_col_stats_filename(col: str) -> str:
    """
    Generates the filename for average sales statistics of a given column.

    Parameters:
        col (str): Column name to generate statistics for (e.g., 'Month', 'DayOfWeek').

    Returns:
        str: Full filename of the CSV file storing average sales statistics.
    """
    filename = f'average_sales_per_{col}.csv'
    return filename


def safe_read_csv(file_path: str) -> pd.DataFrame | None:
    """
    Safely reads a CSV file and handles common file errors.

    Parameters:
        file_path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame | None: Returns DataFrame if read successfully, otherwise None.
    """
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


def get_avg_sales_per_col_df(col: str) -> pd.DataFrame | None:
    """
    Reads the average sales CSV for a given column (Month or DayOfWeek).

    Parameters:
        col (str): Column name for which to get average sales ('Month' or 'DayOfWeek').

    Returns:
        pd.DataFrame | None: DataFrame with average sales or None if file not found.
    """
    full_filename = relative_filepath + get_col_stats_filename(col)
    return safe_read_csv(full_filename)

# -------------------------------
# Streamlit column configuration
# -------------------------------
def get_st_stats_column_config() -> dict:
    """
    Returns a Streamlit column configuration dictionary for displaying
    average sales statistics with proper formatting and help tooltips.

    Columns include:
        - Month: Full month name
        - DayOfWeek: Full weekday name
        - Average_Sales: Average daily sales (formatted to 2 decimals)
        - Total_Open_Days: Total days when store was open
        - Total_Days_With_Promo: Total days with active promotions

    Returns:
        dict: Mapping of column names to Streamlit column_config objects.
    """
    return {
        "Month": st.column_config.TextColumn(
            "Month Name",
            help="Full month name (January, February, ...)"
        ),
        "DayOfWeek": st.column_config.TextColumn(
            "Day Name",
            help="Full weekday name (Monday, Tuesday, ...)"
        ),
        "Average_Sales": st.column_config.NumberColumn(
            "💰 Average Sales",
            help="Average daily sales when the store was open",
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
