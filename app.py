import subprocess 
import sys 
import os

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

def check_and_install_packages():
    # Check if the flag file exists
    flag_file = 'packages_installed.txt'
    if not os.path.exists(flag_file):
        print("First run, installing required packages, please wait...")
        # If the flag file does not exist, install the packages
        packages = ["streamlit-option-menu", "pyarrow", "pyecharts", "streamlit-echarts", "scikit-learn"]
        for package in packages:
            install_package(package)
        
        # Check and install plotly separately
        try:
            import plotly
        except ImportError:
            print("Plotly not found, installing...")
            install_package("plotly")

        # Create the flag file after installation is complete
        with open(flag_file, 'w') as f:
            f.write('Packages installed')
        print("All packages have been successfully installed.")
    else:
        print("Packages have already been installed, skipping installation step.")

# Check and install packages
check_and_install_packages()

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression





##################page settings##################
st.set_page_config(page_title="Car Sales Analysis", page_icon=":bar_chart:", layout="wide")

with st.sidebar:
    selected = option_menu(
        menu_title="Dashboard",
        options=["Overview", "Sales Trend", "Customer","Dealer"],
        default_index=0,
        orientation="vertical",
    )


##################css style settings################

test_css = """
    /*set sidebar background color*/
    section[data-testid='stSidebar'] {
        background-color: #2e2e2e;  /*dark gray background*/
    }

    div[data-testid='stSidebar'] div[class*='option-menu'] {
        background-color: #1e90ff !important;  /*blue background*/
        padding: 10px !important;  /*add some padding*/
        border-radius: 10px !important;  /*rounded corners*/
        margin-bottom: 10px !important;  /*add some margin*/
    }

    /*adjust main container left and right padding*/
    div[data-testid='stAppViewContainer'] {
        padding-right:0.5rem !important;
        padding-left: 0.5rem !important;
    }

    /*hide toolbar*/
    div[data-testid='stToolbar'] {
        display: none !important;
    }

    /*hide plotly chart toolbar logo*/
    div[data-testid='stToolbar'] a.modebar-btn.plotlyjsicon.modebar-btn--logo {
        display: none !important;
    }

    /*hide footer*/
    footer {
        display: none !important;
    }

    /*set entire page background color*/
    div[data-testid='stAppViewContainer'] {
        background-color: black !important;
    }

    /*set all font color to light gray*/
    div[data-testid="stApp"] {
        color: #F3F3F3 !important;
    }

    /*adjust main content area top padding*/
    div[data-testid='block-container'] {
        padding-top: 0rem !important;
    }

    /*set plotly chart style*/
    div[data-testid='stPlotlyChart'] {
        background: #232323 !important;
        padding: 3px !important;
        border-radius: 1rem !important;
        box-shadow: 0.1rem 0.1rem 0.2rem 0 rgba(135,135,135,0.7) !important;
        overflow: hidden; /*ensure content does not overflow container*/
    }

    /*clip chart background to match rounded corners*/
    div[data-testid='stPlotlyChart'] > div {
        clip-path: inset(0 0 0 0 round 1rem);
    }

    /*number display*/
    div[data-testid='stMetric'] {
        background-color: rgba(0, 104, 201, 0.3) !important;
        padding: 3px !important;
        border-radius: 1rem !important;
        box-shadow: 0.1rem 0.1rem 0.2rem 0 rgba(135,135,135,0.7) !important;
        overflow: hidden; /*ensure content does not overflow container*/
        text-align: center; /*center-align numbers*/
    }


    /*clip number display background to match rounded corners*/
    div[data-testid='stMetric'] > div {
        clip-path: inset(0 0 0 0 round 1rem);
    }

    /*hide header*/
    header {
        display: none !important;
    }

    /*hide legend on small screens*/
    @media only screen and (max-width: 550px) {
        g[clip-path*='legend'] {
            display: none !important;
        }
    }

        
    /*set font color to light gray for all elements in streamlit markdown container*/
    div[data-testid="stMarkdownContainer"] * {
        color: #F3F3F3 !important;
    }

    /*set font color to light gray and center-align for streamlit metric value elements*/
    div[data-testid="stMetricValue"] {
        color: #F3F3F3 !important;
        text-align: center !important;
    }

"""


st.markdown(f'<style>{test_css}</style>', unsafe_allow_html=True)



#############import data#################

#read data
df = pd.read_csv("car sales.csv")

#data preprocessing
#convert date format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
df = df.dropna(subset=["Date"])
df['Year'] = df['Date'].dt.year.astype(str)
df['Quarter'] = df['Date'].dt.quarter.astype(str)
df['Month'] = df['Date'].dt.month.astype(str)


#general filters
st.sidebar.header("Filters")
selected_year = st.sidebar.multiselect("Select Year", df["Year"].unique(), default=df["Year"].unique().tolist())
selected_region = st.sidebar.multiselect("Select Dealer Region", df["Dealer_Region"].unique(), default=df["Dealer_Region"].unique().tolist())

#set default values if filters are empty
if not selected_year:
    selected_year = df["Year"].unique().tolist()
if not selected_region:
    selected_region = df["Dealer_Region"].unique().tolist()

#apply filters
filtered_df = df[(df["Year"].isin(selected_year)) & (df["Dealer_Region"].isin(selected_region))]



#########switch to "Overview" page  by TIAN Chen ##############
if selected == "Overview":
    from  Overview_tc import (overview_page)
    overview_page(filtered_df)
    
#########switch to Sales Trend Analysis page by TIAN Chen #########
if selected == "Sales Trend":
    from Sales_Trend_tc import (sales_trend_page)
    sales_trend_page(filtered_df)

#########switch to Customer page by HAN Renruike #############
if selected == "Customer":
    from Customer_hrrk import (customer_page)
    customer_page(filtered_df)  

########switch to Dealer page by ZHU Keye #############
if selected == "Dealer":
     from Dealer_zky import (dealer_page)
     dealer_page(filtered_df) 


