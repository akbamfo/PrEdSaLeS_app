import streamlit as st
import pandas as pd
import pickle
import os
import random
from PIL import Image
from xgboost import XGBRFRegressor

    
# Function to load the ML component
def load_ml_comp(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Header style
header_style = """
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 20px;
"""

# Header at the top of the page
st.markdown(
    f'<div style="{header_style}">'
    f'<h1>WELCOME TO FAVORITA RETAIL STORES</h1>'
    f'<h2>SALES PREDICTION APP</h2>'
    f'<p>This is a simple app for sales prediction to optimize business strategies using accurate sales forecasts. Predict trends, plan inventory, and elevate decision-making.</p>'
    f'</div>',
    unsafe_allow_html=True)


# Specify the path to your image file
image_path = "sales.png"
image = Image.open(image_path)

# Set up the layout with three columns
col1, col2, col3 = st.columns([1, 2, 1])

# Display the image in the middle column with a specific width
image_width = 800  
col2.image(image, caption='Sales Image', width=image_width)

# Creating columns for user inputs and predictions
selection_column, display_column = st.columns(2)

# Header for the selection column
with selection_column:
    st.write("## User Inputs")
    pred_date = st.date_input("Enter the date for your prediction")
    pred_on_promo = st.slider("Select Promo number", min_value=0, max_value=726)

    # Days of the week
    days_of_week = {
        "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
        "Friday": 5, "Saturday": 6, "Sunday": 7
    }

    selected_day_names = st.multiselect("Select specific days", list(days_of_week.keys()))

    # Map selected day names to numeric values
    selected_days = [days_of_week[day_name] for day_name in selected_day_names]

    # Clusters
    cluster_list = list(range(1, 18))
    selected_clusters = st.multiselect("Selected Cluster", cluster_list)

    # Stores
    store_list = list(range(1, 55))
    selected_stores = st.multiselect("Selected Stores", store_list)

    # Product Categories (Dropdown for single select)
    categories = [
        'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
        'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
        'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
        'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
        'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
        'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
        'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
        'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
        'SEAFOOD'
    ]
    selected_category = st.selectbox("Select Category", categories)

# Load ML model
folder_path = r"C:\Users\USER\Azubi LP4\Sales app"
file_name = 'pipeline.pkl'
ml_core_fp = os.path.join(folder_path, file_name)
ml_comp_dict = load_ml_comp(fp=ml_core_fp)

# Display column style
display_column_style = """
    width: 100%;
    padding: 20px;
    border: 1px solid #ccc;
    border-radius: 10px;
"""



# Default prediction value
predictions = 0

if "random_values" not in st.session_state:
    st.session_state.random_values = {
        "pred_saleslag1": round(random.uniform(0, 9), 2),
        "pred_saleslag2": round(random.uniform(0, 9), 2),
        "pred_roll_mean": round(random.uniform(0, 5), 2),
        "pred_roll_std": round(random.uniform(0, 4), 2)
    }

# Button to trigger prediction
# Inside the button click event
if st.button("Predict"):
    # Use the stored random values for input features
    data = {
        "store_nbr": selected_stores,
        "Product": [selected_category],  # Ensure the product category is passed as a list with a single string element
        "onpromotion": [pred_on_promo],
        "cluster": selected_clusters,
        "day_of_week": selected_days,
        "sales_lag_1": [st.session_state.random_values["pred_saleslag1"]],
        "sales_lag_2": [st.session_state.random_values["pred_saleslag2"]],
        "rolling_mean": [st.session_state.random_values["pred_roll_mean"]],
        "rolling_std": [st.session_state.random_values["pred_roll_std"]]
    }
    input_df = pd.DataFrame(data)
    
    # Use the loaded machine learning model for predictions
    predictions = ml_comp_dict.predict(input_df)[0]

# # Display the predictions in the display column
# with display_column:
#     st.markdown(
#         f'<div style="{display_column_style}">'
#         f'<h2>Predictions</h2>'
#         f'<div style="color: {"green" if predictions > 100 else "red"}; font-size: 24px;">{predictions}</div>'
#         f'</div>',
#         unsafe_allow_html=True)

# Format the prediction with two decimal places
formatted_predictions = "{:.2f}".format(predictions)

# Determine the color based on the prediction value
prediction_color = "green" if predictions > 10 else "red"

# Assuming formatted_predictions contains the predicted amount as a string
formatted_predictions = "$" + formatted_predictions

# Display the predictions in the display column with formatting and styling
with display_column:
    st.markdown(
        f'<div style="{display_column_style}">'
        f'<h2>Predictions</h2>'
        f'<div style="color: {prediction_color}; font-size: 24px;">{formatted_predictions}</div>'
        f'</div>',
        unsafe_allow_html=True)


# Set up the layout with three columns
col1, col2, col3 = st.columns([1, 2, 1])
# Designer information
designer_info = "Designed by: Team Zanzibar"

# Team members' names
team_members = ["Kofi Asare Bamfo, Doe Edinam & Enoch Taylor-Nketiah"]

# Display the designer information and team members in the right-bottom corner
col3.markdown(
    f'<div style="position: absolute; bottom: 10px; right: 10px; font-weight: bold;">'
    f'{designer_info}<br/>'
    f'Team Members: {", ".join(team_members)}'
    f'</div>',
    unsafe_allow_html=True)



