import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

def sales_trend_page(filtered_df):

# key indicator - Data Overview
    # Key indicator calculation
    total_sales = filtered_df.shape[0]
    total_revenue = filtered_df['Price ($)'].sum()
    avg_order_revenue = filtered_df['Price ($)'].mean()
    max_price = filtered_df['Price ($)'].max()
    min_price = filtered_df['Price ($)'].min()
    med_price_per_car = filtered_df['Price ($)'].median()

    # show key indicator data
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Sales Volume', f'{total_sales}')
    with col2:
        st.metric('Total Revenue', f'${total_revenue:,.0f}')
    with col3:
        st.metric('Average Order Revenue', f'${avg_order_revenue:,.0f}')

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric('Maxium Price', f'${max_price:,.2f}')
    with col5:
        st.metric('Minimum Price', f'${min_price:,.2f}')
    with col6:
        st.metric('Median Price per Car', f'${med_price_per_car:,.0f}')  
    ############################## Part II：Sales Trend Analysis ##############################

    # Page Title
    st.title('Sales Trend Analysis')

    ############################## fig1-start ##############################

    # Creat Filter: Y/Q/M

    def calculate_sales_data(filtered_df, time_dimension):
    ## Define function， arguments: filtered_df and time_dimension, returns to sales_volume and sales_revenue and x-axis
        
        # Ensure Year column in filtered_df is integer
        filtered_df['Year'] = filtered_df['Year'].astype(str)

        if time_dimension == "Year":  # filtered by year
            # Sales Volume
            sales_volume = filtered_df.groupby('Year').size().reset_index(name = 'Sales Volume')
            sales_volume = sales_volume.sort_values(by = 'Year') # sort by year
            x_col_volume = 'Year'

            # Sales Revenue
            sales_revenue = filtered_df.groupby('Year').agg({'Price ($)': 'sum'}).reset_index()
            sales_revenue = sales_revenue.sort_values(by = 'Year') # sort by year
            sales_revenue['Year'] = sales_revenue['Year'].astype(int) # ensure Year column in sales_revenue is integer
            sales_revenue['Price ($)'] = sales_revenue['Price ($)'].astype(int) # ensure Price ($) column in sales_revenue is integer
            x_col_revenue = 'Year'

        elif time_dimension == "Quarter": # filtered by quarter
            # Sales Volume
            sales_volume = filtered_df.groupby(['Year', 'Quarter']).size().reset_index(name = 'Sales Volume')
            sales_volume['Quarter'] = sales_volume['Year'].astype(str) + " Q" + sales_volume['Quarter'].astype(str)
            sales_volume = sales_volume.sort_values(by = ['Year', 'Quarter']) # sort by year and quarter
            x_col_volume = 'Quarter'

            # Sales Revenue
            sales_revenue = filtered_df.groupby(['Year', 'Quarter']).agg({'Price ($)': 'sum'}).reset_index()
            sales_revenue['Quarter'] = sales_revenue['Year'].astype(str) + " Q" + sales_revenue['Quarter'].astype(str)
            sales_revenue = sales_revenue.sort_values(by=['Year', 'Quarter'])  # sort by year and quarter
            x_col_revenue = 'Quarter' 

        else:   # filtered by month
            # Sales Volume
            sales_volume = filtered_df.groupby(['Year', 'Month']).size().reset_index(name = 'Sales Volume')
            # ensure Year and Month columns in sales_volume are string
            sales_volume['Year'] = sales_volume['Year'].astype(str)
            sales_volume['Month'] = sales_volume['Month'].astype(str)

            sales_volume['Month_num'] = pd.to_datetime(sales_volume['Year'] + '-' + sales_volume['Month'].str.zfill(2)).dt.month
            sales_volume['Month'] = pd.to_datetime(sales_volume['Year'] + '-' + sales_volume['Month'].str.zfill(2)).dt.strftime('%b %Y')
            sales_volume = sales_volume.sort_values(by = ['Year', 'Month_num'])
            sales_volume = sales_volume.drop('Month_num', axis = 1)
            x_col_volume = 'Month'

            # Sales Revenue
            sales_revenue = filtered_df.groupby(['Year', 'Month']).agg({'Price ($)': 'sum'}).reset_index()
            sales_revenue['Year'] = sales_revenue['Year'].astype(str)
            sales_revenue['Month'] = sales_revenue['Month'].astype(str)

            sales_revenue['Month_num'] = pd.to_datetime(sales_revenue['Year'] + '-' + sales_revenue['Month'].str.zfill(2)).dt.month
            sales_revenue['Month'] = pd.to_datetime(sales_revenue['Year'] + '-' + sales_revenue['Month'].str.zfill(2)).dt.strftime('%b %Y')
            sales_revenue = sales_revenue.sort_values(by = ['Year', 'Month_num'])
            sales_revenue = sales_revenue.drop('Month_num', axis=1)
            x_col_revenue = 'Month'

        return sales_volume, sales_revenue, x_col_volume, x_col_revenue  

    time_dimension = st.radio('Select Time Dimension', ['Year', 'Quarter', 'Month'])  # create filter button
    sales_volume, sales_revenue, x_col_volume, x_col_revenue = calculate_sales_data(filtered_df, time_dimension)

    # create combo chart
    fig1 = go.Figure()

    # plot barchart for sales revenue
    fig1.add_trace(go.Bar(x = sales_revenue[x_col_revenue], y = sales_revenue['Price ($)'], 
        name = 'Sales Revenue ($)', marker_color = 'cadetblue'))

    # plot line chart for sales volume
    fig1.add_trace(go.Scatter(x = sales_volume[x_col_volume].astype(str), y = sales_volume['Sales Volume'], 
        name = 'Sales Volume', 
        mode = 'lines+markers', 
        yaxis = 'y2', line = dict(color = 'goldenrod')))

    # set layout parameters
    fig1.update_layout(
            xaxis = dict(title = time_dimension, 
                         tickangle = -45,
                         tickmode = 'array', # set tick mode to array to show
                         tickvals = sales_volume[x_col_volume], # force to use all x-axis values as ticks
                         ticktext = sales_volume[x_col_volume] # use all x-axis labels
            ),
            yaxis = dict(title = 'Sales Revenue ($)', titlefont = dict(color = 'cadetblue')),
            yaxis2 = dict(title = 'Sales Volume', titlefont = dict(color = 'goldenrod'), overlaying = 'y', side = 'right'),
            legend = dict (x = 0.02, y = 0.98),
            barmode = 'group', width = 500, height = 300,
            title_x=0.1,
            title_text="Sales Volume & Revenue Over Time",
            title_font=dict(size=20, color="#F3F3F3"),
            plot_bgcolor="rgba(0, 104, 201, 0)",
            paper_bgcolor="rgba(0, 104, 201, 0.2)",
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20))

    ############################## fig1-end ##############################

    ############################## fig2-start ##############################

    ### sales revenue quarterly comparison 2022 vs 2023

    # uarterly sales revenue calculation
    sales_revenue_comparison = filtered_df[filtered_df['Year'].isin(['2022', '2023'])]
    sales_revenue_comparison = sales_revenue_comparison.groupby(['Year', 'Quarter']).agg({'Price ($)': 'sum'}).reset_index()

    # ensure Quarter in sales_revnue_comparison is integer
    sales_revenue_comparison['Quarter'] = sales_revenue_comparison['Quarter'].astype(int)

    # sort by year and quarter
    sales_revenue_comparison = sales_revenue_comparison.sort_values(by = ['Year', 'Quarter'])

    # calculate the QOQ sales revenue growth rate
    # 1. calculate sales revenue of 2022
    sales_2022 = sales_revenue_comparison[sales_revenue_comparison['Year'] == '2022'].set_index('Quarter')['Price ($)']
    # 2. calculate sales revenue of 2023
    sales_2023 = sales_revenue_comparison[sales_revenue_comparison['Year'] == '2023'].set_index('Quarter')['Price ($)']
    # 3. calculate growth rate
    growth_rate = ((sales_2023-sales_2022) / sales_2022) * 100

    # plot bar chart
    fig2 = px.bar(sales_revenue_comparison, x = 'Quarter', y = 'Price ($)',
        color = 'Year',
        labels = {'Price ($)': 'Sales Revenues($)', 'Quarter': 'Quarter'},
        barmode = 'group',
        color_discrete_map = {'2022': 'cadetblue', '2023': 'goldenrod'})

    # set layout parameters
    fig2.update_layout(width = 500, height = 300,
            title_x=0.1,
            title_text="Sales Volume & Revenue Over Time",
            title_font=dict(size=20, color="#F3F3F3"),
            plot_bgcolor="rgba(0, 104, 201, 0)",
            paper_bgcolor="rgba(0, 104, 201, 0.2)",
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)           
        ) 

    # add annotation of growth rate
    for quarter in sales_2023.index:
        rate = growth_rate.get(quarter)
        if rate is not None:
            fig2.add_annotation(x = quarter, y = sales_2023[quarter],
            text = f'{rate:.1f}%', showarrow = False, yshift = 10)

    ############################## fig2-end ##############################

    # show fig1 and fig2 side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1)  # fig1
        

    with col2:
        st.plotly_chart(fig2)  # fig2
        

    ############################## fig3-start ##############################

    ### add subtitle
    st.subheader("Monthly Sales Revenue Deep Dive")

    # rename column 'Company' to 'Brand'
    filtered_df = filtered_df.rename(columns={'Company': 'Brand'})

    # add filter options
    col1, col2, col3, col4 = st.columns(4)

    # nultiple filter for color and brand
    selected_colors = col1.multiselect('Select Color', options = sorted(filtered_df['Color'].dropna().unique().tolist()), default = [])
    selected_brands = col2.multiselect('Select Brand', options = sorted(filtered_df['Brand'].dropna().unique().tolist()), default = [])

    # single filter for transmission type
    selected_transmission = col3.selectbox('Select Transmission', options=['All'] + sorted(filtered_df['Transmission'].dropna().unique().tolist()))

    # accord Model type when Brand is choosed
    if selected_brands:
        model_options = sorted(filtered_df[filtered_df['Brand'].isin(selected_brands)]['Model'].dropna().unique().tolist())
    else:
        model_options = sorted(filtered_df['Model'].dropna().unique().tolist())

    selected_models = col4.multiselect('Select Model', options = model_options, default = [])

    # applied filter
    filtered_data = filtered_df.copy()

    if selected_colors:
        filtered_data = filtered_data[filtered_data['Color'].isin(selected_colors)]

    if selected_brands:
        filtered_data = filtered_data[filtered_data['Brand'].isin(selected_brands)]

    if selected_transmission != 'All':
        filtered_data = filtered_data[filtered_data['Transmission'] == selected_transmission]

    if selected_models:
        filtered_data = filtered_data[filtered_data['Model'].isin(selected_models)]

    # show total sales revenue after filter
    sales_revenue = filtered_data['Price ($)'].sum()
    st.write(f'### Total Sales Revenue: ${sales_revenue:,.0f}')

    # calculate monthly sales revenue
    monthly_sales = filtered_data.groupby('Month')['Price ($)'].sum().reset_index()

    # sort by month and ensure the type is integer, make sure x-axis output is correct and in order
    monthly_sales['Month'] = monthly_sales['Month'].astype(int)
    monthly_sales = monthly_sales.sort_values(by = 'Month')

    # convert Month type to string to apply in color
    monthly_sales['Month'] = monthly_sales['Month'].astype(str)

    # plot bar chart
    fig3 = px.bar(monthly_sales, x = 'Month', y = 'Price ($)',
        color = 'Month',
        color_discrete_sequence = px.colors.qualitative.Pastel)

    # set layout parameters
    fig3.update_layout(
        xaxis = dict(
            type = 'category',  # force x-axis to be the categorize axis
            tickmode = 'array',  # use array to set scale
            tickvals = sorted(monthly_sales['Month'].unique()),  
            ticktext = sorted(monthly_sales['Month'].unique())),
        width = 1000, height = 400,
        plot_bgcolor="rgba(0, 104, 201, 0)",
        paper_bgcolor="rgba(0, 104, 201, 0.2)",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig3) 
            

    ############################# fig3-end ###################################
