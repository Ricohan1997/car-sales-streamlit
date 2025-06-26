import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np


def overview_page(filtered_df):
    # Welcome title and purpose of this App
    st.title('Welcome to the Car Sales Analysis App')
    st.write("""
    - This App is used to analyze car sales data in US through dealers to help users understand sales trends, customer preferences, and market distribution.
    - Through interactive charts and tables, users can deep dive into the data and draw insights.
    """)

    # read data
    df = pd.read_csv("car sales.csv")

    # data overview
    st.header('Data Overview')
    st.dataframe(df.head(3)) # show the first 3 lines as sample

    # data cleaning and sanity check
    with st.expander('Data Cleaning and Sanity Checking'):
        st.subheader('Missing Value Check')

        # Replace empty strings with NaN
        df['Customer Name'] = df['Customer Name'].replace('', pd.NA)

        # Clean invisible characters
        df['Customer Name'] = df['Customer Name'].str.strip().replace(r'[\n\t\r]', '', regex=True)

        # Ensure column is string type
        df['Customer Name'] = df['Customer Name'].astype(str)

        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0]) # only show the missing columns

        # check the number of gender
        gender_counts = df['Gender'].value_counts()
        st.write("Unique Values in 'Gender' column:")
        st.write(gender_counts)  

        # use histogram to check outliers in numerical columns
        st.write('Outliers in Numerical Columns')
        num_columns = ['Annual Income', 'Price ($)']
        fig, axes = plt.subplots(1, len(num_columns), figsize = (15, 5))
        for i, col in enumerate(num_columns):
            sns.histplot(df[col], kde = True, ax = axes[i])
            axes[i].set_title(f'Histogram of {col}')
        st.pyplot(fig)

 
    
    