import streamlit as st
import pandas as pd
import joblib
import json
import altair as alt
import numpy as np



custom_css = """"""
# --- Page Configuration ---
st.set_page_config(
    page_title="Rental Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ASSET LOADING ---
# Use st.cache_resource to load the model and data only once
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, feature list, and location data."""
    # Load the ML model
    model = joblib.load('house_price_model.pkl')

    # Load the list of model features
    with open('model_features.json', 'r') as f:
        model_features = json.load(f)

    # Load the location data (for the dropdown)
    with open('locations.json', 'r') as f:
        locations = json.load(f)

     # Load the location data (for the dropdown)
    with open('location_sorted.json', 'r') as f:
        location_sorted = json.load(f)

    return model, model_features, locations, location_sorted

# Call the function to load the assets
model, model_features, locations, location_sorted = load_assets()

@st.cache_data
def load_dataframe():
    """Loads the simulated housing data."""
    df = pd.read_csv('df_houses_new.csv')
    return df


df_houses = load_dataframe()

# --- APP LAYOUT ---
st.set_page_config(layout="wide") # Use the full page width

# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Add a sidebar for a professional touch
# Add a sidebar for a professional touch
with st.sidebar:
    st.header("About This App ℹ️")
    st.write("""
    This app demonstrates an interactive front-end for a house price prediction model. The focus is on a user-friendly design.

    **Technology Used:**
    - **Frontend:** Streamlit
    - **Visualization:** Plotly
    - **ML Model:** Scikit-learn (Random Forest)
    """)


st.title("🏡 Rental Price Predictor")

# --- PREDICTION LOGIC ---
def predict_price_category(location, unit_type, outdoor_space, flooring, parking, laundry, ac, dishwasher, stainless, cable, internet, pool, fitness):
    """Preprocesses user input and returns the predicted price category."""

    # 1. Location to Coordinates Mapping
    latitude, longitude = locations[location]

    # 2. Unit Type to Bedrooms Mapping
    # Extracts the number from strings like "1 bedroom" or sets 0 for "Studio"
    if unit_type.startswith("Studio"):
        bedrooms = 0
    else:
        bedrooms = int(unit_type.split()[0])

    # 3. Create a dictionary for the raw feature values
    # This mirrors the original DataFrame's structure
    feature_dict = {
        'Parking': 'Off-street parking' if parking == 'Yes' else 'not',
        'Flooring': 'Hardwood' if flooring == 'Wood' else ('Carpet' if flooring == 'Carpet' else 'other'),
        'Laundry': 'In-unit' if laundry == 'Yes' else 'not',
        'Dishwasher': 'Available' if dishwasher == 'Yes' else 'not',
        'Stainless_Appliances': 'Available' if stainless == 'Yes' else 'not',
        'Cable': 'Available' if cable == 'Yes' else 'not',
        'Internet': 'Available' if internet == 'Yes' else 'not',
        'Outdoor_Spaces': outdoor_space,
        'AC': 'Central' if ac == 'Yes' else 'not',
        'Pool': 'Available' if pool == 'Yes' else 'not',
        'Fitness_Facilities': 'Available' if fitness == 'Yes' else 'not',
        'rental_type': 'Monthly',  # Hard-coded as decided
        'Bedrooms': bedrooms,
        'latitude': latitude,
        'longitude': longitude
    }

    # 4. Create a DataFrame from the dictionary
    input_df = pd.DataFrame([feature_dict])

    # 5. One-Hot Encode the categorical features
    # This converts text categories into the numerical format the model needs
    input_encoded = pd.get_dummies(input_df)

    # 6. Align columns with the model's training features
    # This is a crucial step! It ensures the app's input has the exact same columns
    # as the data the model was trained on, filling any missing ones with 0.
    input_aligned = input_encoded.reindex(columns=model_features, fill_value=0)

    # 7. Make the prediction
    prediction = model.predict(input_aligned)

    # 8. Return the result
    return prediction[0]

# --- UI LAYOUT ---
st.markdown("---")

# Create tabs for different sections of the app
tab1, tab2, tab3 = st.tabs(["**Predictor 🔮**", "**Data Insights 📊**", "**Model Details ⚙️**"])

with tab1:
    st.header("Get Your Price Prediction")
    st.write("Fill in the details below. The model will predict the price category for a rental property.")

    with st.form("prediction_form"):
        col1, col2 = st.columns([2, 1]) # Make the first column wider

        with col1:
            st.subheader("Property Details")

            location = st.selectbox("📍 Location", options=sorted(location_sorted.keys()))

            unit_type = st.selectbox(
                "🏡 Unit Type",
                options=["Studio", "1 bedroom", "2 bedroom", "3 bedroom", "4 bedroom", "5 bedroom"]
            )

            outdoor_space = st.selectbox(
                "🌳 Outdoor Space",
                options=["No Outdoor Space", "Balcony", "Patio", "Combined"]
            )

            flooring = st.selectbox(
                "🪵 Flooring",
                options=["Wood", "Carpet", "Tile", "Other"]
            )

        with col2:
            st.subheader("Amenities")
            # Use st.expander to make the form compact
            with st.expander("Optional Amenities", expanded=True):
                parking = st.selectbox("Parking?", ("Yes", "No"), key='park')
                laundry = st.selectbox("Laundry?", ("Yes", "No"), key='laund')
                ac = st.selectbox("AC?", ("Yes", "No"), key='ac')
                dishwasher = st.selectbox("Dishwasher?", ("Yes", "No"), key='dish')
                stainless = st.selectbox("Stainless Steel?", ("Yes", "No"), key='ss')
                cable = st.selectbox("Cable?", ("Yes", "No"), key='cable')
                internet = st.selectbox("Internet?", ("Yes", "No"), key='inet')
                pool = st.selectbox("Pool?", ("Yes", "No"), key='pool')
                fitness = st.selectbox("Fitness Center?", ("Yes", "No"), key='fit')

        # Submit button for the form
        submitted = st.form_submit_button("Predict Price Category")
    # --- HANDLE FORM SUBMISSION AND DISPLAY RESULT ---
    if submitted:
        # Show a spinner while the prediction is being made
        with st.spinner('🧠 Analyzing features and predicting...'):
            # Call the prediction function with the user's input
            prediction_result = predict_price_category(
                location, unit_type, outdoor_space, flooring, parking, laundry, ac,
                dishwasher, stainless, cable, internet, pool, fitness
            )

        # Define the price categories and engaging descriptions
        price_category_map = {
            0: ("Low Range", "Below $975", "This property falls into an affordable price bracket. A great value find! 💰"),
            1: ("Mid Range", "$975 - $1325", "This property is competitively priced for the average market. A solid choice! 👍"),
            2: ("High Range", "$1325 - $1750", "This property is in the premium range, likely offering better amenities or location. ✨"),
            3: ("Luxury Range", "Above $1750", "This is a top-tier property, indicating luxury features and prime location. 💎")
        }

        # Get the details from our map
        category_name, price_range, description = price_category_map[prediction_result]

        st.markdown("---")
        st.subheader("Prediction Result")

        st.markdown(f"""
            <style>
            .metric-container {{
                background-color: #262730; /* Match your dark theme background */
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
            }}
            .metric-value {{
                font-size: 2.5rem;
                font-weight: bold;
                color: #FFFFFF; /* White text */
            }}
            .metric-delta {{
                background-color: #00A36C; /* Green background for the pill */
                color: #FFFFFF;
                border-radius: 20px;
                padding: 0.2rem 0.6rem;
                font-size: 1rem;
                font-weight: bold;
                display: inline-block;
                margin-top: 1rem;
            }}
            </style>

            <div class="metric-container">
                <div class="metric-value">{category_name}</div>
                <div class="metric-delta">↑ {price_range}</div>
            </div>
            """, unsafe_allow_html=True)

        # Use st.metric for a visually appealing output
        #st.metric(label="Predicted Price Category", value=category_name, delta=price_range)

        st.info(description) # Use st.info for a nice blue box
    

with tab2:

   # --- Theme & Styling ---
    alt.themes.enable("dark")
    st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #393939;
        text-align: center;
        padding: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Data Loading (Cached) ---
    @st.cache_data
    def load_data():
        """Loads the housing data and renames columns for st.map()."""
        df = pd.read_csv('housing.csv')
        df = df.dropna()
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        return df

    df_original = load_data()
    AMENITIES = ['Parking', 'Laundry', 'Dishwasher', 'Stainless_Appliances', 'Cable','Internet', 'Outdoor_Spaces', 'AC', 'Pool', 'Fitness_Facilities']


    # --- Main Dashboard Title ---
    st.title("Comprehensive Housing Market Analysis")
    st.markdown("An all-in-one view of rental listings, prices, amenities, and locations.")

    st.divider()

    # ==================================================================================================
    # Section 1: At a Glance Overview
    # ==================================================================================================
    st.header("🏠 At a Glance Overview")

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total Listings", value=f"{len(df_original):,}")
    col2.metric(label="Average Price", value=f"${df_original['price'].mean():,.0f}")
    col3.metric(label="Median Price", value=f"${df_original['price'].median():,.0f}")
    col4.metric(label="Number of Cities", value=f"{df_original['city'].nunique()}")

    # --- Top Cities and State Listings Charts ---
    col1, col2 = st.columns((2, 1.5), gap='medium')
    with col1:
        st.markdown('#### Top 5 Most Expensive Cities')
        df_top_cities = df_original.groupby('city')['price'].mean().nlargest(5).sort_values(ascending=True).reset_index()
        chart_top_cities = alt.Chart(df_top_cities).mark_bar(cornerRadius=5).encode(
            x=alt.X('price:Q', title='Average Price', axis=alt.Axis(format='$,.0f')),
            y=alt.Y('city:N', title='City', sort=None),
            tooltip=['city', alt.Tooltip('price:Q', format='$,.0f')]
        ).properties(height=300)
        st.altair_chart(chart_top_cities, use_container_width=True)

    with col2:
        st.markdown('#### Listings per State')
        df_state_counts = df_original['state'].value_counts().reset_index()
        chart_state_listings = alt.Chart(df_state_counts).mark_bar().encode(
            x=alt.X('state:N', title='State', sort='-y'),
            y=alt.Y('count:Q', title='Number of Listings'),
            tooltip=['state', 'count']
        ).properties(height=300)
        st.altair_chart(chart_state_listings, use_container_width=True)

    st.divider()

    # ==================================================================================================
    # Section 2: Deep Dive into Price Analysis
    # ==================================================================================================
    st.header("💰 Deep Dive into Price Analysis")

    # --- Price Analysis Filters ---
    price_col1, price_col2 = st.columns(2)
    with price_col1:
        min_price, max_price = int(df_original['price'].min()), int(df_original['price'].max())
        price_range = st.slider(
            'Filter by price range',
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )
    with price_col2:
        bedroom_list = sorted(df_original['bedrooms'].unique())
        selected_bedrooms = st.multiselect('Filter by number of bedrooms', bedroom_list, default=bedroom_list)

    # --- Filter data for Price Analysis Section ---
    df_price_filtered = df_original[
        (df_original['price'] >= price_range[0]) &
        (df_original['price'] <= price_range[1]) &
        (df_original['bedrooms'].isin(selected_bedrooms))
    ]

    # --- Price Analysis Charts ---
    st.markdown('### Price Distribution & Features')
    if df_price_filtered.empty:
        st.warning("No data available for the selected price filters.")
    else:
        price_chart_col1, price_chart_col2 = st.columns(2, gap="medium")
        with price_chart_col1:
            st.write('**Price Distribution**')
            hist_chart = alt.Chart(df_price_filtered).mark_bar().encode(
                alt.X('price:Q', bin=alt.Bin(maxbins=50), title='Price ($)'),
                alt.Y('count()', title='Number of Listings')
            ).properties(height=350)
            st.altair_chart(hist_chart, use_container_width=True)
        
        with price_chart_col2:
            st.write('**Price by Number of Bedrooms**')
            box_plot_bedrooms = alt.Chart(df_price_filtered).mark_boxplot(extent='min-max').encode(
                x=alt.X('bedrooms:O', title='Number of Bedrooms'),
                y=alt.Y('price:Q', title='Price ($)', axis=alt.Axis(format='$,.0f'))
            ).properties(height=350)
            st.altair_chart(box_plot_bedrooms, use_container_width=True)

    st.divider()

    # ==================================================================================================
    # Section 3: Amenities & Features Explorer
    # ==================================================================================================
    st.header("✨ Amenities & Features Explorer")
    amenity_col1, amenity_col2 = st.columns((1.5, 2), gap="medium")

    with amenity_col1:
        st.markdown('#### Most Common Amenities')
        amenity_counts = df_original[AMENITIES].mean().sort_values(ascending=False) * 100
        amenity_df = amenity_counts.reset_index()
        amenity_df.columns = ['amenity', 'percentage']
        chart_amenities = alt.Chart(amenity_df).mark_bar().encode(
            x=alt.X('percentage:Q', title='Frequency (%)'),
            y=alt.Y('amenity:N', title='Amenity', sort='-x')
        ).properties(height=350)
        st.altair_chart(chart_amenities, use_container_width=True)

    with amenity_col2:
        st.markdown('#### Price Impact of an Amenity')
        amenity_to_analyze = st.selectbox('Select an amenity to analyze its price impact', AMENITIES)
        df_original['has_amenity'] = np.where(df_original[amenity_to_analyze], f'With {amenity_to_analyze.title()}', f'Without {amenity_to_analyze.title()}')
        chart_price_impact = alt.Chart(df_original).mark_boxplot(extent='min-max').encode(
            x=alt.X('has_amenity:N', title='', sort='-y'),
            y=alt.Y('price:Q', title='Price ($)', axis=alt.Axis(format='$,.0f')),
            color='has_amenity:N'
        ).properties(height=350)
        st.altair_chart(chart_price_impact, use_container_width=True)

    st.divider()

    # ==================================================================================================
    # Section 4: Interactive Map View
    # ==================================================================================================
    st.header("🗺️ Interactive Map View")

    # --- Map Filters ---
    map_filter_col1, map_filter_col2, map_filter_col3 = st.columns(3)
    with map_filter_col1:
        city_list = ['All'] + sorted(df_original['city'].unique())
        selected_city = st.selectbox('Filter by city', city_list)
    with map_filter_col2:
        map_bedroom_list = ['All'] + sorted(df_original['bedrooms'].unique())
        selected_map_bedrooms = st.selectbox('Filter by bedrooms', map_bedroom_list, key="map_bedrooms") # Key prevents widget conflict
    with map_filter_col3:
        # Use a different key for the map price slider to avoid conflict with the one in the price analysis section
        map_price_range = st.slider(
            'Filter by price range',
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
            key="map_price_slider"
        )

    # --- Filter data for Map Section ---
    df_map_filtered = df_original[
        (df_original['price'] >= map_price_range[0]) & (df_original['price'] <= map_price_range[1])
    ]
    if selected_city != 'All':
        df_map_filtered = df_map_filtered[df_map_filtered['city'] == selected_city]
    if selected_map_bedrooms != 'All':
        df_map_filtered = df_map_filtered[df_map_filtered['bedrooms'] == selected_map_bedrooms]
        
    # --- Display Map and Data ---
    st.markdown(f"Showing **{len(df_map_filtered)}** properties on the map based on your filters.")
    if df_map_filtered.empty:
        st.warning("No properties to display on the map for the selected filters.")
    else:
        st.map(df_map_filtered, zoom=3)
        with st.expander("View Filtered Data for Map"):
            st.dataframe(
                df_map_filtered[['price', 'city', 'state', 'bedrooms', 'unit_type']],
                hide_index=True,
                use_container_width=True
            )

with tab3:
    st.header("About the Model")
    st.write("""
    This application uses a **Random Forest Classifier** model, a powerful machine learning algorithm, to predict the price category of a rental property based on its features.
    """)

    st.subheader("How was it trained? 🤔")
    st.write("""
    The model was trained on a dataset of thousands of rental listings from across the United States. Key features include location coordinates, property type, and a variety of common amenities.
    """)

    st.subheader("Performance Metrics 🚀")
    st.write("These metrics are from the model's performance on a held-out test dataset, which it had never seen before.")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**Accuracy: 89.1%**")
        st.write("Accuracy measures how many predictions the model got right out of all predictions. An accuracy of 89.1% means the model correctly predicted the price category for nearly 9 out of every 10 properties.")

    with col2:
        st.info("**Macro F1-Score: 0.89**")
        st.write("The F1-Score is a balanced measure of a model's precision (how many selected items are relevant) and recall (how many relevant items are selected). A score of 0.89 indicates a strong and balanced performance across all price categories.")

st.markdown("---")