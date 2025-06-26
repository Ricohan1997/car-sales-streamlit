

def dealer_page(filtered_df):
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import seaborn as sbn
    import matplotlib.pyplot as plt

            
    ###############pic1 dealer sales volume################
    def plot_dealer_sales(filtered_df):

        #calculate dealer sales volume
        dealer_sales = filtered_df.groupby('Dealer_Name').size().reset_index(name='Sales')

        #sort
        dealer_sales = dealer_sales.sort_values('Sales', ascending=True)
        top10_dealer = dealer_sales.head(10)

        top10_dealer['shorter_name'] = top10_dealer['Dealer_Name'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)

        #create bar chart
        fig_dealer = px.bar(
            top10_dealer,
            y='shorter_name', 
            x='Sales', 
            orientation='h', 
            title='TOP10 Dealer Sales Performance',
            labels={'shorter_name': 'Dealer Name', 'Sales': 'Sales Count'},
            text_auto=True,  #show sales volume on bars
            color='Dealer_Name',  #assign different colors to each dealer
            color_discrete_sequence=px.colors.qualitative.Prism  #set color sequence
        )


        #set chart layout
        fig_dealer.update_layout(
            showlegend=False,  #remove legend
            plot_bgcolor="rgba(0, 104, 201, 0.2)",  #plot area background color
            paper_bgcolor="rgba(0, 104, 201, 0.2)",  #entire canvas background color
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.1,
            title_font=dict(size=20, color="#F3F3F3"),
            height = 300,
            xaxis_title=None,
            yaxis_title=None
        )

        #show chart
        st.plotly_chart(fig_dealer, use_container_width=True)


    ############ pic2 car brand and model #############
    def plot_company_sales(filtered_df):

        #calculate sales data groupby company and model
        model_sales = filtered_df.groupby(['Company', 'Model'], as_index=False).size()
        model_sales = model_sales.rename(columns={'size': 'model_nb'})

        #calculate total sales for each company
        company_total_sales = model_sales.groupby('Company', as_index=False)['model_nb'].sum()
        company_total_sales = company_total_sales.rename(columns={'model_nb': 'Total_Sales'})

        # find top 10
        top10_company = company_total_sales.head(10)

        #merge total sales with original data and sort by number
        model_sales = model_sales.merge(top10_company, on='Company')
        model_sales = model_sales.sort_values(['Total_Sales', 'model_nb'], ascending=[False, False])

        #set colors
        unique_companies = model_sales['Company'].unique()
        color_palette = px.colors.qualitative.Set3
    
        #set colors
        company_color_map = {}
        for i, company in enumerate(unique_companies):
            color_index = i % len(color_palette)
            color = color_palette[color_index]
            company_color_map[company] = color

        #create chart
        fig_company = px.bar(
            model_sales,
            y='Company', 
            x='model_nb', 
            color='Company',  
            orientation='h', 
            title='TOP10 Company Model Sales Scale', 
            labels={'model_nb': 'Sales Count', 'Company': 'Company'}, 
            text_auto=True,  #show sales numbers
            text='Model',  # show model names on the bars
            hover_data=['Model'],  # show model names on hover
            color_discrete_map=company_color_map 
        )

        #add black border around bars
        fig_company.update_traces(
            marker=dict(line=dict(width=1, color='black'))
        )

        #set layout
        fig_company.update_layout(
            showlegend=False,  #remove legend
            plot_bgcolor="rgba(0, 104, 201, 0.2)",
            paper_bgcolor="rgba(0, 104, 201, 0.2)",  
            margin=dict(l=20, r=20, t=40, b=20),
            title_x=0.1,
            title_font=dict(size=20, color="#F3F3F3"),
            height = 300,
            xaxis_title=None, #remove x title
            yaxis_title=None  #remove y title
        )

        #show chart
        st.plotly_chart(fig_company, use_container_width=True)


    #############pic3 map and bar chart #####################
    def map_region_sales(filtered_df):
        #map city to state
        city_to_state = {
            'Middletown': 'OH',  
            'Aurora': 'IL',     
            'Greenville': 'SC',  
            'Pasco': 'WA',     
            'Janesville': 'WI',
            'Scottsdale': 'AZ', 
            'Austin': 'TX'     
        }
        
        #calculate sales for each region
        region_sales = filtered_df.groupby('Dealer_Region').size().reset_index(name='Sales')
        region_sales['State'] = region_sales['Dealer_Region'].map(city_to_state)
        
        #sort sales by volume in ascending order
        region_sales_sorted = region_sales.sort_values('Sales', ascending=True)
        

        #set color
        color_mapping = {
        'AZ': '#8D6E63',  
        'IL': '#607D8B',  
        'OH': '#6D8B74',  
        'SC': '#78909C',  
        'TX': '#C5B9AC',  
        'WA': '#556B2F',  
        'WI': '#BCAAA4'   
        }
        #set layout to display map and bar chart side by side
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, 
            cols=2,
            specs=[[{"type": "choropleth"}, {"type": "bar"}]],
            column_widths=[0.6, 0.4]
        )
        
        #add map to first column
        choropleth = px.choropleth(
            region_sales,
            locations='State',
            locationmode='USA-states',
            color='State', 
            scope='usa',
            color_discrete_map=color_mapping,  #use custom color mapping
            labels={'Sales': 'Sales Count', 'State': 'State'},
            hover_name='Dealer_Region',
            hover_data={'Sales': True},  #show hover data
        )
        
        #set trace
        for trace in choropleth.data:
            fig.add_trace(trace, row=1, col=1)
        
        #add bar chart to second column
        bar = px.bar(
            region_sales_sorted,
            y='Dealer_Region',
            x='Sales',
            orientation='h',
            text_auto=True,
            color='State', 
            color_discrete_map=color_mapping  #use the same color mapping
        )
        bar.update_traces(width=0.6)
        
        for trace in bar.data:
            fig.add_trace(trace, row=1, col=2)
        
        #set up map
        fig.update_geos(
            scope='usa',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            countrycolor='rgb(255, 255, 255)',
            lakecolor='rgb(255, 255, 255)',
            subunitcolor='rgb(255, 255, 255)'
        )
        
        #set up overall layout
        fig.update_layout(
            title_x=0.4,
            title_text="Regional Sales Map",
            title_font=dict(size=20, color="#F3F3F3"),
            showlegend=False,
            height=300,
            plot_bgcolor="rgba(0, 104, 201, 0)",
            paper_bgcolor="rgba(0, 104, 201, 0.2)",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        #show chart
        st.plotly_chart(fig, use_container_width=True)



    ##################pic4 line chart####################
    def line_region_sales(filtered_df):  
       
        ##regional market share
        #parse date, extract year and month
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        filtered_df['YearMonth'] = filtered_df['Date'].dt.to_period('M').astype(str)  #convert to "YYYY-MM" format

        #calculate sales by month and dealer region
        region_monthly_sales = filtered_df.groupby(['YearMonth', 'Dealer_Region']).size().reset_index(name='Sales')

        #calculate sales per month
        monthly_total_sales = region_monthly_sales.groupby('YearMonth')['Sales'].transform('sum')

        #calculate sales share
        region_monthly_sales['SalesShare'] = (region_monthly_sales['Sales'] / monthly_total_sales) * 100

        #set line fig
        fig_line = px.line(
            region_monthly_sales,
            x='YearMonth',
            y='SalesShare', 
            color='Dealer_Region',  
            markers=True,
            labels={'YearMonth': 'Month', 'SalesShare': 'Sales Share (%)', 'Dealer_Region': 'Region'},
        )

        #set layout
        fig_line.update_layout(
            title_x=0.3,
            title_text="Monthly Sales Share(%) by Dealer Region",
            title_font=dict(size=20, color="#F3F3F3"),
            yaxis_title="Sales Share (%)",  #change y-axis title
            legend_title="Dealer Region",
            legend=dict(font=dict(color="#7c8192")),
            plot_bgcolor="rgba(0, 104, 201, 0.2)",  #plot area background color
            paper_bgcolor="rgba(0, 104, 201, 0.2)",  #entire canvas background color
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )

        #display chart
        st.plotly_chart(fig_line, use_container_width=True)


    ######################  page layout  #####################
    
    #find top selling dealer
    top_selling_dealer = filtered_df['Dealer_Name'].value_counts().idxmax()
    #find top selling region
    top_selling_region = filtered_df['Dealer_Region'].value_counts().idxmax()
    #count unique company brand
    unique_company = filtered_df['Company'].nunique()
    #count dealer 
    unique_dealers = filtered_df['Dealer_Name'].nunique()

    #show three numbers
    col1, col2, col3 , col4 = st.columns(4)
    with col1:
        st.metric('Top Selling Dealer', f'{top_selling_dealer[:11]}')
    with col2:
        st.metric('Top Selling Region', f'{top_selling_region}')
    with col3:
         st.metric('Dealer Brand Count', f'{unique_company}')
    with col4:
        st.metric('Number of Dealers', f'{unique_dealers}')


    #show dealer and company sales
    col_dealer_sales,col_company_sales = st.columns(2)
    with col_dealer_sales:
        plot_dealer_sales(filtered_df)
    with col_company_sales:
        plot_company_sales(filtered_df)


    #show map
    map_region_sales(filtered_df)

    #show line chart
    line_region_sales(filtered_df)



