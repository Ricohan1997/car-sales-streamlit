import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

def sales_trend_page(filtered_df):
    # Key indicator calculation
    total_sales = filtered_df.shape[0]
    total_revenue = filtered_df['Price ($)'].sum()
    avg_order_revenue = filtered_df['Price ($)'].mean()
    max_price = filtered_df['Price ($)'].max()
    min_price = filtered_df['Price ($)'].min()
    med_price_per_car = filtered_df['Price ($)'].median()

    # Display metrics
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

    st.title('Sales Trend Analysis')

    # Helper function to group data
    def calculate_sales_data(filtered_df, time_dimension):
        filtered_df['Year'] = filtered_df['Year'].astype(str)

        if time_dimension == "Year":
            sales_volume = filtered_df.groupby('Year').size().reset_index(name='Sales Volume')
            sales_volume = sales_volume.sort_values(by='Year')
            x_col_volume = 'Year'

            sales_revenue = filtered_df.groupby('Year').agg({'Price ($)': 'sum'}).reset_index()
            sales_revenue = sales_revenue.sort_values(by='Year')
            sales_revenue['Year'] = sales_revenue['Year'].astype(int)
            sales_revenue['Price ($)'] = sales_revenue['Price ($)'].astype(int)
            x_col_revenue = 'Year'

        elif time_dimension == "Quarter":
            sales_volume = filtered_df.groupby(['Year', 'Quarter']).size().reset_index(name='Sales Volume')
            sales_volume['Quarter'] = sales_volume['Year'].astype(str) + " Q" + sales_volume['Quarter'].astype(str)
            sales_volume = sales_volume.sort_values(by=['Year', 'Quarter'])
            x_col_volume = 'Quarter'

            sales_revenue = filtered_df.groupby(['Year', 'Quarter']).agg({'Price ($)': 'sum'}).reset_index()
            sales_revenue['Quarter'] = sales_revenue['Year'].astype(str) + " Q" + sales_revenue['Quarter'].astype(str)
            sales_revenue = sales_revenue.sort_values(by=['Year', 'Quarter'])
            x_col_revenue = 'Quarter'

        else:  # Month
            sales_volume = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Sales Volume')
            sales_volume['Year'] = sales_volume['Year'].astype(str)
            sales_volume['Month'] = sales_volume['Month'].astype(str)
            sales_volume['Month_num'] = pd.to_datetime(sales_volume['Year'] + '-' + sales_volume['Month'].str.zfill(2)).dt.month
            sales_volume['Month'] = pd.to_datetime(sales_volume['Year'] + '-' + sales_volume['Month'].str.zfill(2)).dt.strftime('%b %Y')
            sales_volume = sales_volume.sort_values(by=['Year', 'Month_num']).drop('Month_num', axis=1)
            x_col_volume = 'Month'

            sales_revenue = filtered_df.groupby(['Year', 'Month']).agg({'Price ($)': 'sum'}).reset_index()
            sales_revenue['Year'] = sales_revenue['Year'].astype(str)
            sales_revenue['Month'] = sales_revenue['Month'].astype(str)
            sales_revenue['Month_num'] = pd.to_datetime(sales_revenue['Year'] + '-' + sales_revenue['Month'].str.zfill(2)).dt.month
            sales_revenue['Month'] = pd.to_datetime(sales_revenue['Year'] + '-' + sales_revenue['Month'].str.zfill(2)).dt.strftime('%b %Y')
            sales_revenue = sales_revenue.sort_values(by=['Year', 'Month_num']).drop('Month_num', axis=1)
            x_col_revenue = 'Month'

        return sales_volume, sales_revenue, x_col_volume, x_col_revenue

    time_dimension = st.radio('Select Time Dimension', ['Year', 'Quarter', 'Month'])
    sales_volume, sales_revenue, x_col_volume, x_col_revenue = calculate_sales_data(filtered_df, time_dimension)

    # --------- fig1 ---------
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=sales_revenue[x_col_revenue],
        y=sales_revenue['Price ($)'],
        name='Sales Revenue ($)',
        marker_color='cadetblue'
    ))
    fig1.add_trace(go.Scatter(
        x=sales_volume[x_col_volume].astype(str),
        y=sales_volume['Sales Volume'],
        name='Sales Volume',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='goldenrod')
    ))

    fig1.update_layout(
        xaxis=dict(
            title=time_dimension,
            tickangle=-45,
            tickmode='array',
            tickvals=sales_volume[x_col_volume].tolist(),  # ← FIX HERE
            ticktext=sales_volume[x_col_volume].tolist()   # ← AND HERE
        ),
        yaxis=dict(title='Sales Revenue ($)', titlefont=dict(color='cadetblue')),
        yaxis2=dict(title='Sales Volume', titlefont=dict(color='goldenrod'), overlaying='y', side='right'),
        legend=dict(x=0.02, y=0.98),
        barmode='group', width=500, height=300,
        title_x=0.1,
        title_text="Sales Volume & Revenue Over Time",
        title_font=dict(size=20, color="#F3F3F3"),
        plot_bgcolor="rgba(0, 104, 201, 0)",
        paper_bgcolor="rgba(0, 104, 201, 0.2)",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # --------- fig2 ---------
    sales_revenue_comparison = filtered_df[filtered_df['Year'].isin(['2022', '2023'])]
    sales_revenue_comparison = sales_revenue_comparison.groupby(['Year', 'Quarter']).agg({'Price ($)': 'sum'}).reset_index()
    sales_revenue_comparison['Quarter'] = sales_revenue_comparison['Quarter'].astype(int)
    sales_revenue_comparison = sales_revenue_comparison.sort_values(by=['Year', 'Quarter'])

    sales_2022 = sales_revenue_comparison[sales_revenue_comparison['Year'] == '2022'].set_index('Quarter')['Price ($)']
    sales_2023 = sales_revenue_comparison[sales_revenue_comparison['Year'] == '2023'].set_index('Quarter')['Price ($)']
    growth_rate = ((sales_2023 - sales_2022) / sales_2022) * 100

    fig2 = px.bar(
        sales_revenue_comparison, x='Quarter', y='Price ($)',
        color='Year',
        labels={'Price ($)': 'Sales Revenues($)', 'Quarter': 'Quarter'},
        barmode='group',
        color_discrete_map={'2022': 'cadetblue', '2023': 'goldenrod'}
    )

    fig2.update_layout(
        width=500, height=300,
        title_x=0.1,
        title_text="Quarterly Revenue Comparison",
        title_font=dict(size=20, color="#F3F3F3"),
        plot_bgcolor="rgba(0, 104, 201, 0)",
        paper_bgcolor="rgba(0, 104, 201, 0.2)",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    for quarter in sales_2023.index:
        rate = growth_rate.get(quarter)
        if rate is not None:
            fig2.add_annotation(
                x=quarter,
                y=sales_2023[quarter],
                text=f'{rate:.1f}%',
                showarrow=False,
                yshift=10
            )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

