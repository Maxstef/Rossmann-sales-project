import streamlit as st
from src.sales_statistics import get_avg_sales_per_col_df, get_st_stats_column_config
from src.predict_inputs_view import render_daily_predict_form_features
# from src.predict_helper import predict_daily todo

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
st.markdown(
    """
    The sections below present key statistics on **average daily sales**  
    based on historical data from **2013–2014**.

    > 💡 **Note:** 
    > All statistics are calculated **only for days when stores are open**.  
    > For instance, most stores are closed on Sundays, but the reported average sales represent only the cases **when stores were actually open**.
    """,
    unsafe_allow_html=True
)

# Create two columns with padding between them
col1, col_, col2 = st.columns([1, 0.05, 1])  # second column acts as spacing

# --- Left column: by Month ---
with col1:
    st.subheader("📅 By Month")
    st.dataframe(
        get_avg_sales_per_col_df('Month'),
        hide_index=True,
        use_container_width=True,
        column_config=get_st_stats_column_config()
    )

# --- Right column: by Day of Week ---
with col2:
    st.subheader("🗓️ By Day of Week")
    st.dataframe(
        get_avg_sales_per_col_df('DayOfWeek'),
        hide_index=True,
        use_container_width=True,
        column_config=get_st_stats_column_config()
    )

# Add a horizontal divider for better visual separation
st.markdown("---")

# -------------------------------
#  Predict
# -------------------------------
st.header("Daily Sales Prediction")

user_input = render_daily_predict_form_features()

if st.button('Predict Sales'):
    st.markdown("### User Data (JSON)")
    st.json(user_input)

    # todo
    # prediction = predict_daily()
    # st.markdown(prediction)

