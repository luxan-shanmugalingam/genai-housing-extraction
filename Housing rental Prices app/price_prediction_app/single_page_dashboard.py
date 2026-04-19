import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="US Housing Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    df = pd.read_csv('data/housing_data.csv')
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    return df

df_original = load_data()
AMENITIES = ['parking', 'pool', 'ac', 'gym', 'balcony']

# --- Sidebar ---
with st.sidebar:
    st.title("🏠 US Housing Dashboard")
    st.info("This dashboard combines all analyses into a single view. Use the sections below to explore the data.")
    with st.expander('About this app', expanded=True):
        st.write('''
            - **Data Source**: A synthetic dataset generated for demonstration.
            - **Purpose**: To provide a comprehensive, interactive tool for analyzing housing rental data across various US cities.
        ''')


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