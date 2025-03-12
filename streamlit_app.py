import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import plotly.express as px

from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Added missing import


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

    st.markdown("Group members: Nada Beltagui, Deema Hazim, Breanna Richard, and Kanishk Surana.")
    
    st.success("This project leverages machine learning and interactive dashboards to provide deeper insights into Singapore’s HDB resale market trends.")
elif selected == "📊 Data Exploration":
    st.markdown("### 🔍 Data Exploration")
    
    # User selection: Head or Tail
    view_option = st.radio("View dataset: ", ["Head", "Tail"], horizontal=True)
    num = st.number_input("Select number of rows to view", 5, 20)
    
    if view_option == "Head":
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))
    
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

    st.title("🏠 Singapore Resale Prices Dashboard")
    st.markdown("Explore insights from resale prices dataset.")

    # Sidebar filters
    st.sidebar.header("Filter Data")
    year_range = st.sidebar.slider("Select Year Range", int(df["year"].min()), int(df["year"].max()), (2015, 2023))
        
    filtered_flat_types = df["flat_type"].unique()
    filtered_flat_types = [ft for ft in filtered_flat_types if ft not in ["MULTI-GENERATION", "EXECUTIVE"]]

    flat_type = st.sidebar.multiselect("Select Flat Type", filtered_flat_types, filtered_flat_types)
    
    # Apply filters
    filtered_df = df[(df["year"].between(year_range[0], year_range[1])) & (df["flat_type"].isin(flat_type))]

    ### 📈 Resale Price Trends Over Time
    st.subheader("📈 Resale Price Trends Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=filtered_df, x="year", y="resale_price", hue="flat_type", marker="o", ax=ax)
    plt.xlabel("Year")
    plt.ylabel("Average Resale Price (SGD)")
    plt.title("Resale Price Trends Over Time")
    st.pyplot(fig)

    ### 🏙️ Average Resale Price by Town
    st.subheader("🏙️ Average Resale Price by Town")
    st.markdown("""
    This bar chart compares the **average resale prices** across different towns in Singapore.  
    - **Bukit Timah and Bukit Panjang** have the highest resale prices.  
    - **Tao Payoh and Woodlands Yishun** have the lowest prices.  
    - This allows buyers to compare affordability across different locations.
    """)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=filtered_df.groupby("town")["resale_price"].mean().reset_index(), 
                x="town", y="resale_price", color="steelblue", ax=ax)
    plt.xlabel("Town")
    plt.ylabel("Average Resale Price (SGD)")
    plt.xticks(rotation=45)
    plt.title("Average Resale Price by Town")
    st.pyplot(fig)

    ### 📈 Price vs Floor Area (Scatter Plot)
    st.subheader("📈 Price vs Floor Area (Scatter Plot)")
    st.markdown("""
    This scatter plot visualizes the relationship between **floor area (sqm)** and **resale price (SGD)** for properties in Singapore.  
    - Each dot represents a property listing.  
    - The red **line of best fit** shows the general trend: **larger flats tend to have higher resale prices**.
    """)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=filtered_df, x="floor_area_sqm", y="resale_price", scatter_kws={"alpha": 0.5}, line_kws={"color": "red"}, ax=ax)
    plt.xlabel("Floor Area (sqm)")
    plt.ylabel("Resale Price (SGD)")
    plt.title("Resale Price vs Floor Area (with Line of best fit)")
    st.pyplot(fig)

    ### 🔗 Correlation Matrix
    st.subheader("🔗 Correlation Matrix")
    st.markdown("""
    The correlation matrix helps us understand relationships between different numerical features.  
    - A value close to **1** or **-1** indicates a strong correlation.  
    - **Darker colors** show stronger correlations.
    """)
    
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    excluded_columns = ["latitude", "longitude", "postal_code"]
    st.write(numeric_columns)

    filtered_numeric_columns = [col for col in numeric_columns if col not in excluded_columns]

# Let the user select only from the filtered columns
    selected_vars = st.multiselect("Select variables for correlation matrix", filtered_numeric_columns, default=filtered_numeric_columns)

# sns.heatmap(correlation)
    if len(selected_vars)>1:
        tab_corr, = st.tabs(["Correlation"])
        correlation = df[selected_vars].corr()
        fig = px.imshow(correlation.values, x=correlation.index, y=correlation.columns, labels=dict(color="Correlation"), text_auto=".2f")
        tab_corr.plotly_chart(fig, theme = "streamlit", use_container_width=True)


    ### 🗺 Interactive Map of Resale Transactions
    st.subheader("🗺 HDB Resale Locations in Singapore")
    st.markdown("""
    This map visualizes the location of the homes in the dataset in Singapore based on latitude and longitude.  
    - Each point represents a home.  
    - You can zoom in to explore different areas.
    """)

    # You can also customize the width, height, and scrolling
    st.components.v1.iframe("https://lookerstudio.google.com/embed/reporting/f6bfd06d-2009-4e2a-9826-51787d8af70e/page/m659E", width=800, height=500, scrolling=True)

# Prediction section
elif selected == "🤖 Prediction":
    
    # Title for Prediction Section
    st.markdown("### 🤖 Price Prediction Using Machine Learning")

    ## Step 1: Split dataset into X (features) and y (target variable)
    x = df[["closest_mrt_dist", "floor_area_sqm", "number_of_rooms"]]  # Features
    y = df["resale_price"]  # Target variable

    # Split data into training and testing sets (80% training, 20% testing)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    ## Step 2: Train the Linear Regression model
    linear = LinearRegression()
    linear.fit(x_train, y_train)  # Train model on training data

    ## Step 3: Make predictions
    predictions = linear.predict(x_test)

    ## Step 4: Calculate error metrics
    mae = mean_absolute_error(y_test, predictions)  # Mean Absolute Error
    mse = mean_squared_error(y_test, predictions)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error

    ## Step 5: User selection for output
    option = st.selectbox(
        "Choose what to display:",
        ["Metrics (MAE, MSE, RMSE)", "Actual vs Predicted Plot", "Predict a Price"]
    )

    # Display error metrics
    if option == "Metrics (MAE, MSE, RMSE)":
        st.write("📉 *Mean Absolute Error (MAE):*", round(mae, 2))
        st.write("📈 *Mean Squared Error (MSE):*", round(mse, 2))
        st.write("📊 *Root Mean Squared Error (RMSE):*", round(rmse, 2))

    # Display scatter plot of actual vs predicted values
    elif option == "Actual vs Predicted Plot":
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=predictions)  # Scatter plot
        plt.plot(
            [y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            color='red', linestyle='--'
        )  # Reference line for perfect prediction
        
        plt.xlabel("Actual Resale Price")
        plt.ylabel("Predicted Resale Price")
        plt.title("Actual vs. Predicted Resale Prices")

        st.pyplot(plt)  # Show plot in Streamlit

    elif option == "Predict a Price":
        st.write("Enter the property details below to get a predicted price:")

        # Let user enter inputs
        closest_mrt_dist_input = st.number_input(
            "Closest MRT distance (meters):",
            min_value=0.0,
            max_value=50000.0,
            value=500.0,
            step=50.0,
        )

        floor_area_input = st.number_input(
            "Floor area (sqm):",
            min_value=10.0,
            max_value=500.0,
            value=80.0,
            step=5.0,
        )

        number_of_rooms_input = st.number_input(
            "Number of rooms:",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

        # Put inputs into an array/dataframe for prediction
        user_input = np.array([[ closest_mrt_dist_input, floor_area_input, number_of_rooms_input]])

        # Make prediction
        predicted_price = linear.predict(user_input)

        # Show the predicted price
        st.subheader("Predicted Resale Price:")
        st.write(f"${predicted_price[0]:,.2f}")