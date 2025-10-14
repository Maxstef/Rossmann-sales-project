import pandas as pd

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

