import streamlit as st
from src.sales_statistics import get_avg_sales_per_col_df
# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Rossmann Sales",
    page_icon="📊",
    layout="wide",  # Makes use of full screen width
    initial_sidebar_state="expanded"
)

# -------------------------------
# App title and introduction
# -------------------------------
st.title("📈 Rossmann Sales Dashboard")

st.markdown(
    """
    **Welcome!**  

    This app helps you:
    - Predict expected sales for each store  
    - Plan product inventory efficiently  
    - Manage your budget and forecast revenue  

    Use the sidebar to select filters and explore sales statistics.
    """,
    unsafe_allow_html=True
)

# Add a horizontal divider for better visual separation
st.markdown("---")

# -------------------------------
# Statistical overview
# -------------------------------
st.header("Sales Statistics Overview")

# Create two columns
col1, col2 = st.columns(2)

# --- Left column: by Month ---
with col1:
    st.subheader("By Month")
    st.dataframe(
        get_avg_sales_per_col_df('Month'),
        hide_index=True
    )

# --- Right column: by Day of Week ---
with col2:
    st.subheader("By Day of Week")
    st.dataframe(
        get_avg_sales_per_col_df('DayOfWeek'),
        hide_index=True
    )

# Add a horizontal divider for better visual separation
st.markdown("---")

# -------------------------------
#  Predict
# -------------------------------
st.header("Daily Sales Prediction")

# TODO 