import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from streamlit_option_menu import option_menu

st.set_page_config(page_title="🏡 Singapore HDB Resale Analysis", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f7f9fc;}
        .stTabs [data-baseweb="tab-list"] button {background-color: #d1e7ff; color: black; border-radius: 8px; margin: 5px;}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {background-color: #0d6efd; color: white;}
        .stMarkdown h3 {color: #0d6efd;}
    </style>
""", unsafe_allow_html=True)

# Navigation menu
selected = option_menu(menu_title=None, options=["📖 Description", "📊 Data Exploration", "📈 Visualization", "🤖 Prediction"], default_index=0, orientation="horizontal")

df = pd.read_csv("singapore_cleaned.csv")

if selected == "📖 Description":
    st.title("🏡 Singapore HDB Resale Price Analysis and Prediction")
    
    st.markdown("""
    ### 🔍 **Introduction**
    The Singaporean housing market is a dynamic and evolving landscape, particularly within the **Housing and Development Board (HDB) resale segment**. 
    As Singapore continues to grow, understanding historical price trends and developing predictive models for future housing prices is crucial for both policymakers and potential homeowners.
    """)
    
    st.markdown("""
    ### 📂 **Dataset Overview**
    - **Source:** [Kaggle’s Singapore HDB Resale Prices (1990-2023)](https://www.kaggle.com/datasets/talietzin/singapore-hdb-resale-prices-1990-to-2023?select=resale-flat-prices-based-on-registration-date-from-jan-2017-onwards_locationdata.csv)
    - **Filtered Data:** Only resale transactions from **January 2017 onwards**
    - **Modifications:**
      - Removed incomplete rows
      - Retained only `flat_type` entries in "X-room" format
      - Added `number_of_rooms` column for better predictions
    
    📌 [View Data Preprocessing in Colab](https://colab.research.google.com/drive/15fF1Ev1ez6jv89FtoFhFcSBmH-ZUhwWL?usp=sharing)
    """)
    
    st.markdown("""
    ### 🎯 **Project Goals**
    1. **Exploratory Data Analysis (EDA):** Identify key trends and correlations.
    2. **Data Visualization:** Interactive dashboards with **Looker**.
    3. **Predictive Modeling:** Machine learning-based price forecasting.
    """)
    
    st.success("This project leverages machine learning and interactive dashboards to provide deeper insights into Singapore’s HDB resale market trends.")
elif selected == "📊 Data Exploration":
    st.markdown("### 🔍 Data Exploration")
    
    # Display dataset
    num = st.number_input("Select number of rows to view", 5, 20)
    st.dataframe(df.head(num))
    
    # Summary statistics
    st.markdown("#### 📊 Summary Statistics")
    st.dataframe(df.drop(columns=["latitude", "longitude", "postal_code"]).describe())    
    
    # Dataset shape
    st.write("📏 **Dataset Shape:**", df.shape)

    # Display column data types
    st.markdown("#### ⚙️ Data Types")
    st.write(df.dtypes)

    # Unique values per column
    st.markdown("#### 🔢 Unique Values Per Column")
    st.write(df.nunique())

elif selected == "📈 Visualization":
    st.markdown("### 📊 Data Visualization")
    
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_vars = st.multiselect("Select variables for correlation matrix", numeric_columns, default=numeric_columns[:3])
    
    if len(selected_vars) > 1:
        tab_corr, = st.tabs(["📊 Correlation Matrix"])
        correlation = df[selected_vars].corr()
        fig = px.imshow(correlation.values, x=correlation.index, y=correlation.columns, labels=dict(color="Correlation"), color_continuous_scale="RdBu_r")
        tab_corr.plotly_chart(fig, theme="streamlit", use_container_width=True)

elif selected == "🤖 Prediction":
    st.markdown("### 🤖 Price Prediction Using Machine Learning")
    
    from sklearn.model_selection import train_test_split  
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    ## Step 1: split dataset into X and y
    x = df.drop(columns=[""])
    y = df[""]
    
    ## Step 2: split between train and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    ## Step 3: initialize the linear regression model
    linear = LinearRegression()
    
    ## Step 4: training the model
    linear.fit(x_train, y_train)
    
    ## Step 5: make predictions
    predictions = linear.predict(x_test)
    
    ## Step 6: evaluate model performance
    mae = mean_absolute_error(predictions, y_test)
    
    ## Display results
    st.write("📉 **Mean Absolute Error (MAE):**", round(mae, 2))
    st.write("📈 **Sample Predictions:**")
    st.dataframe(pd.DataFrame({"Actual": y_test.values[:10], "Predicted": predictions[:10]}))