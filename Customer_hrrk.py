import streamlit as st
import pandas as pd
import plotly.express as px
from pyecharts.charts import Pie
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts

def customer_page(filtered_df):
    # =================== 4️⃣ Key Metrics ===================
    # Create four columns to display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Display the top-selling brand based on the highest frequency in the 'Company' column
    col1.metric('Top Selling Brand', filtered_df['Company'].value_counts().idxmax())
    
    # Display the top-selling model based on the highest frequency in the 'Model' column
    col2.metric('Top Selling Model', filtered_df['Model'].value_counts().idxmax())
    
    # Display the top-selling color based on the highest frequency in the 'Color' column
    col3.metric('Top Selling Color', filtered_df['Color'].value_counts().idxmax())
    
    # Display the number of unique customers based on the unique values in the 'Customer Name' column
    col4.metric('Number of Unique Customers', filtered_df['Customer Name'].nunique())

   # =================== 5️⃣ Pie Chart and Bar Chart in Columns ===================
    col1, col2 = st.columns(2)

    # Pie Chart: Color Market Share
    with col1:
        color_counts = filtered_df['Color'].value_counts()
        color_data = [list(i) for i in zip(color_counts.index, color_counts)]
        
        # color mapping
        color_map = {
            'Pale White': '#f3f3f3',  
            'Black': '#2c2c2c',       
            'Red': '#ff5f5f'          
        }
        
        pie_chart = (
            Pie()
            .add(
                "Color Share",
                color_data,
                radius=["40%", "75%"],
                label_opts=opts.LabelOpts(
                    formatter="{b}: {d}%",
                    position="outside",
                    font_size=12,
                    color="#f3f3f3"
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Color Market Share",
                    pos_top="5%",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=16,
                        color="#f3f3f3"
                    )
                ),
                legend_opts=opts.LegendOpts(
                    pos_left="left",
                    orient="vertical",
                    pos_top="middle",
                    textstyle_opts=opts.TextStyleOpts(color="white")
                )
            )
        )
        
        # Set color
        for i, (color_name, _) in enumerate(color_data):
            pie_chart.options.get('series')[0].get('data')[i]['itemStyle'] = {
                'color': color_map.get(color_name, '#808080')  
            }

        st_pyecharts(pie_chart, height="300px")


    # Bar Chart: Brand Sales Comparison
    with col2:
        # Get the top 10 brands with the highest sales count
        company_counts = filtered_df['Company'].value_counts().head(10).reset_index()
        company_counts.columns = ['Company', 'Sales Count']
        
        # Create a bar chart to compare brand sales
        fig_bar = px.bar(
            company_counts,
            x='Company',
            y='Sales Count',
            color='Company',
            color_discrete_sequence=px.colors.qualitative.Plotly  # Set the color sequence for the bars
        )

        # Add a black border to the bars for better visualization
        fig_bar.update_traces(marker=dict(line=dict(width=1, color='black')))

        fig_bar.update_layout(
            height=300,  # Set the height of the bar chart
            title_x=0.2,  # Center the title horizontally
            title_y=0.95,  # Set the vertical position of the title
            title_text="Brand Sales Comparison",  # Set the title text
            title_font=dict(size=20, color="#F3F3F3"),  # Set title font size and color
            plot_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the plot background color
            paper_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the paper background color
            showlegend=False,  # Hide the legend for the bar chart
            margin=dict(l=20, r=20, t=60, b=20)  # Adjust margins around the chart
        )
        st.plotly_chart(fig_bar, use_container_width=True)  # Display the bar chart

    # =================== 6️⃣ Price Analysis ===================
    # Create a box plot to show the price distribution by brand
    fig_box = px.box(filtered_df, x="Company", y="Price ($)", color="Company")
    fig_box.update_layout(
        height=400,  # Set the height of the box plot
        title_x=0.35,  # Center the title horizontally
        title_y=0.95,  # Set the vertical position of the title
        title_text="Price Distribution by Brand",  # Set the title text
        title_font=dict(size=20, color="#F3F3F3"),  # Set title font size and color
        plot_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the plot background color
        paper_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the paper background color
        showlegend=False,  # Hide the legend for the box plot
        margin=dict(l=20, r=20, t=60, b=20)  # Adjust margins around the chart
    )
    st.plotly_chart(fig_box, use_container_width=True)  # Display the box plot

    # =================== 7️⃣ Heatmap Analysis ===================
    # Create a heatmap to show the relationship between Gender and Body Style preferences
    heatmap_data = filtered_df.groupby(["Gender", "Body Style"]).size().reset_index(name="Count")
    fig_heatmap = px.density_heatmap(
        heatmap_data,
        x="Body Style",
        y="Gender",
        z="Count",  # The count of each gender-body style combination
        color_continuous_scale="Blues"  # Set the color scale for the heatmap
    )
    fig_heatmap.update_layout(
        height=300,  # Set the height of the heatmap
        title_x=0.35,  # Center the title horizontally
        title_y=0.95,  # Set the vertical position of the title
        title_text="Gender vs. Body Style Preference",  # Set the title text
        title_font=dict(size=20, color="#F3F3F3"),  # Set title font size and color
        plot_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the plot background color
        paper_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the paper background color
        showlegend=False  # Hide the legend for the heatmap
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)  # Display the heatmap

       # =================== 8⃣ Income-Price Analysis ==================
    # Generate a scatter plot 
    try:
        fig_scatter = px.scatter(
            filtered_df,
            x="Annual Income",  # X-axis represents the annual income of customers
            y="Price ($)",  # Y-axis represents the price of the car purchased
            hover_data=["Customer Name", "Company", "Model"],  # Show customer details on hover
            color_discrete_sequence=["#1f77b4"]  # Set the color for the scatter points
        )
        
        # Customize the layout of the scatter plot
        fig_scatter.update_layout(
            height=500,  # Set the plot height
            title_x=0.4,  # Center the title horizontally
            title_y=0.95,  # Set the vertical position of the title
            title_text="Income vs Price",  # Set the plot title
            title_font=dict(size=20, color="#F3F3F3"),  # Configure title font size and color
            plot_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the background color of the plot area
            paper_bgcolor="rgba(0, 104, 201, 0.2)",  # Set the background color of the entire figure
            xaxis=dict(
                title="Annual Income ($)",  # Label for the X-axis
                tickprefix="$",  # Add a dollar sign prefix to tick values
                gridcolor="rgba(255,255,255,0.2)"  # Light gridlines for better readability
            ),
            yaxis=dict(
                title="Price ($)",  # Label for the Y-axis
                tickprefix="$",  # Add a dollar sign prefix to tick values
                gridcolor="rgba(255,255,255,0.2)"  # Light gridlines for better readability
            ),
            margin=dict(l=20, r=20, t=80, b=20)  # Adjust plot margins
        )
        
        # Display the scatter plot in the Streamlit app
        st.plotly_chart(fig_scatter, use_container_width=True)

    except Exception as e:
        # Display an error message if an exception occurs
        st.error(f"Analysis error: {str(e)}")
