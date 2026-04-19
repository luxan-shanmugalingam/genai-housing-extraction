# app_shap.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Rental Price Category — Predictor + SHAP",
    page_icon="🏡",
    layout="wide"
)

# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("house_price_model.pkl")
    with open("model_features.json", "r") as f:
        model_features = json.load(f)
    with open("locations.json", "r") as f:
        locations = json.load(f)
    return model, model_features, locations

@st.cache_data
def load_dataframe():
    try:
        df = pd.read_csv("df_houses.csv")
    except Exception:
        df = pd.DataFrame()
    return df

model, model_features, locations = load_assets()
df_houses = load_dataframe()

# ---------------------------
# Helpers
# ---------------------------
def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce everything to numeric, NaN→0, cast to float."""
    return df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

UNIT_TYPES = [
    "Studio (0 bedroom)", "1 bedroom", "2 bedroom", "3 bedroom", "4 bedroom", "5 bedroom"
]
OUTDOOR_SPACE_OPTS = ["No Outdoor Space", "Balcony", "Patio", "Combined"]
FLOORING_OPTS = ["Wood", "Carpet", "Tile", "Other"]

PRICE_CATEGORY_MAP = {
    0: ("Low Range", "Budget-friendly"),
    1: ("Mid Range", "Average market range"),
    2: ("High Range", "Upper market range"),
    3: ("Premium/Luxury", "Top-tier pricing")
}

def _bedrooms_from_unit(unit_type: str) -> int:
    if unit_type.startswith("Studio"):
        return 0
    return int(unit_type.split()[0])

def _yn_to_pair(value_yes_no: str, yes_value: str, no_value: str) -> str:
    """Map 'Yes'/'No' to tokens the model was trained on."""
    return yes_value if value_yes_no == "Yes" else no_value

def _cable_internet_token(value_yes_no: str) -> str:
    return "included" if value_yes_no == "Yes" else "not"

def _fitness_token(value_yes_no: str) -> str:
    # model_features show 'Fitness_Facilities_available' vs '..._not'
    return "available" if value_yes_no == "Yes" else "not"

def transform_inputs_to_model_row(
    location: str,
    unit_type: str,
    outdoor_space: str,
    flooring_ui: str,
    parking_yn: str,
    laundry_yn: str,
    ac_yn: str,
    dishwasher_yn: str,
    stainless_yn: str,
    cable_yn: str,
    internet_yn: str,
    pool_yn: str,
    fitness_yn: str
) -> tuple[pd.DataFrame, dict]:
    """Create a single-row DataFrame matching the model's expected features."""
    # 1) latitude/longitude from locations.json
    latitude, longitude = locations[location]

    # 2) bedrooms from unit type
    bedrooms = _bedrooms_from_unit(unit_type)

    # 3) flooring tokens to match model_features: wood/tile/carpet/other
    if flooring_ui == "Wood":
        flooring = "wood"
    elif flooring_ui == "Tile":
        flooring = "tile"
    elif flooring_ui == "Carpet":
        flooring = "carpet"
    else:
        flooring = "other"

    # 4) raw feature dict (tokens must match training one-hots)
    feature_dict = {
        "latitude": latitude,
        "longitude": longitude,
        "Parking": _yn_to_pair(parking_yn, "provided", "not"),
        "Flooring": flooring,                       # wood/tile/carpet/other
        "Laundry": _yn_to_pair(laundry_yn, "provided", "not"),
        "Dishwasher": _yn_to_pair(dishwasher_yn, "provided", "not"),
        "Stainless_Appliances": _yn_to_pair(stainless_yn, "provided", "not"),
        "Cable": _cable_internet_token(cable_yn),   # included/not
        "Internet": _cable_internet_token(internet_yn),  # included/not
        "Outdoor_Spaces": outdoor_space,            # Balcony/Combined/No Outdoor Space/Patio
        "AC": _yn_to_pair(ac_yn, "provided", "not"),
        "Pool": _yn_to_pair(pool_yn, "provided", "not"),
        "Fitness_Facilities": _fitness_token(fitness_yn),  # available/not
        "rental_type": "Monthly",                   # training included _Monthly vs _other
        "Bedrooms": bedrooms                        # will be one-hot to Bedrooms_0..5
    }

    raw_df = pd.DataFrame([feature_dict])

    # One-hot encode + align to model features
    aligned = pd.get_dummies(raw_df)
    aligned = aligned.reindex(columns=model_features, fill_value=0)

    # Ensure latitude/longitude exist if one-hot dropped them
    if "latitude" in model_features and "latitude" not in aligned.columns:
        aligned["latitude"] = latitude
    if "longitude" in model_features and "longitude" not in aligned.columns:
        aligned["longitude"] = longitude

    # Fill any missing expected columns
    for col in model_features:
        if col not in aligned.columns:
            aligned[col] = 0

    # Reorder and force numeric
    aligned = aligned[model_features]
    aligned = _ensure_numeric(aligned)
    return aligned, feature_dict

# ---------------------------
# SHAP utilities
# ---------------------------
@st.cache_data
def _background_matrix(df_houses: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Build a background (reference) matrix for SHAP by transforming df_houses
    with a similar encoding used for the single-row input.
    """
    if df_houses.empty:
        return pd.DataFrame([np.zeros(len(model_features))], columns=model_features)

    df_bg = df_houses.copy()

    # Map Yes/No -> provided/not for available columns
    for col in ["Parking", "Laundry", "AC"]:
        if col in df_bg.columns:
            df_bg[col] = df_bg[col].map({"Yes": "provided", "No": "not"}).fillna("not")

    # Bedrooms from Unit_Type (if present)
    if "Unit_Type" in df_bg.columns:
        def _bed(u):
            s = str(u)
            if s.startswith("Studio"):
                return 0
            try:
                return int(s.split()[0])
            except Exception:
                return 0
        df_bg["Bedrooms"] = df_bg["Unit_Type"].apply(_bed)

    # Rental type
    df_bg["rental_type"] = "Monthly"

    # Latitude/Longitude from locations.json using 'Location'
    if "Location" in df_bg.columns:
        def _lat(loc):
            try:
                return locations[str(loc)][0]
            except Exception:
                return 0.0
        def _lon(loc):
            try:
                return locations[str(loc)][1]
            except Exception:
                return 0.0
        df_bg["latitude"] = df_bg["Location"].apply(_lat)
        df_bg["longitude"] = df_bg["Location"].apply(_lon)

    keep_cols = [
        c for c in df_bg.columns if c in
        ["latitude", "longitude", "Parking", "Laundry", "AC", "rental_type", "Bedrooms"]
    ]
    X_bg_raw = df_bg[keep_cols]

    X_bg = pd.get_dummies(X_bg_raw)
    X_bg = X_bg.reindex(columns=model_features, fill_value=0)
    X_bg = _ensure_numeric(X_bg)

    if len(X_bg) > 200:
        X_bg = X_bg.sample(200, random_state=0)
    return X_bg

# IMPORTANT: leading underscores so Streamlit won't try to hash unhashable args (the model)
@st.cache_resource
def _get_explainer(_model, _X_bg):
    import shap
    return shap.Explainer(_model, _X_bg)

def _plot_shap_for_prediction(model, model_features, X_row, prediction_result):
    """
    Compute and display a SHAP explanation for a single-row input X_row.
    Handles multiclass by selecting the predicted class.
    """
    import shap
    import matplotlib.pyplot as plt

    # Ensure numeric inputs
    X_row = _ensure_numeric(X_row)

    X_bg = _background_matrix(df_houses, model_features)
    explainer = _get_explainer(model, X_bg)

    shap_values = explainer(X_row)

    # Build the right Explanation object depending on output type
    if hasattr(model, "classes_") and model.classes_ is not None and len(getattr(model, "classes_", [])) > 1:
        classes = list(model.classes_)
        pred_idx = classes.index(prediction_result)

        # shap_values can be a list (old) or a single Explanation (new API)
        if isinstance(shap_values, list):
            values = shap_values[pred_idx][0]
            base_vec = getattr(explainer, "expected_value", [0.0] * len(classes))
            base_value = base_vec[pred_idx] if isinstance(base_vec, (list, tuple, np.ndarray)) else float(base_vec)
        else:
            values = shap_values.values[0, :, pred_idx]
            base = getattr(shap_values, "base_values", None)
            if base is not None and hasattr(base, "ndim") and base.ndim == 2:
                base_value = base[0, pred_idx]
            elif base is not None and hasattr(base, "ndim") and base.ndim == 1:
                base_value = base[pred_idx]
            else:
                ev = getattr(explainer, "expected_value", [0.0] * len(classes))
                base_value = ev[pred_idx] if isinstance(ev, (list, tuple, np.ndarray)) else float(ev)

        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=X_row.iloc[0].values,
            feature_names=list(X_row.columns)
        )
    else:
        # Binary/regression fallbacks
        explanation = shap_values[0] if isinstance(shap_values, shap._explanation.Explanation) else shap_values

    st.subheader("Why did the model predict this? (SHAP) 🧩")
    fig = plt.figure()
    try:
        shap.plots.waterfall(explanation, max_display=12, show=False)
    except Exception:
        shap.plots.bar(explanation, max_display=12, show=False)
    st.pyplot(fig, clear_figure=True)
    st.caption(
        "Positive SHAP values push the prediction **toward** the shown class (or higher price), "
        "while negative values push **away**. The base value is the model’s baseline before seeing your inputs."
    )

# ---------------------------
# UI — Predictor
# ---------------------------
st.title("🏡 Rental Price Category — Predictor + SHAP")

with st.form("predict-form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        location = st.selectbox("📍 Location", options=sorted(list(locations.keys())))
        unit_type = st.selectbox("🛏️ Unit Type", options=UNIT_TYPES)
        outdoor_space = st.selectbox("🌳 Outdoor Space", options=OUTDOOR_SPACE_OPTS)
        flooring = st.selectbox("🪵 Flooring", options=FLOORING_OPTS)

    with col2:
        parking = st.selectbox("🚗 Parking", options=["Yes", "No"])
        laundry = st.selectbox("🧺 In-unit Laundry", options=["Yes", "No"])
        ac = st.selectbox("❄️ AC (Central/Provided)", options=["Yes", "No"])
        dishwasher = st.selectbox("🍽️ Dishwasher", options=["Yes", "No"])

    with col3:
        stainless = st.selectbox("🔧 Stainless Appliances", options=["Yes", "No"])
        cable = st.selectbox("📺 Cable Included", options=["Yes", "No"])
        internet = st.selectbox("🌐 Internet Included", options=["Yes", "No"])
        pool = st.selectbox("🏊 Pool", options=["Yes", "No"])
        fitness = st.selectbox("💪 Fitness Facilities", options=["Yes", "No"])

    submitted = st.form_submit_button("Predict Price Category")

if submitted:
    with st.spinner("🧠 Analyzing features and predicting..."):
        X_row, raw_features = transform_inputs_to_model_row(
            location=location,
            unit_type=unit_type,
            outdoor_space=outdoor_space,
            flooring_ui=flooring,
            parking_yn=parking,
            laundry_yn=laundry,
            ac_yn=ac,
            dishwasher_yn=dishwasher,
            stainless_yn=stainless,
            cable_yn=cable,
            internet_yn=internet,
            pool_yn=pool,
            fitness_yn=fitness
        )
        # Prediction
        y_pred = model.predict(X_row)[0]
        # Optional: probability view if available
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_row)[0]
                y_prob = dict(zip(getattr(model, "classes_", range(len(proba))), proba))
            except Exception:
                pass

    # Display result
    label, desc = PRICE_CATEGORY_MAP.get(int(y_pred), (f"Class {y_pred}", ""))
    st.success(f"**Predicted Price Category:** {label}")
    if desc:
        st.write(desc)
    if y_prob is not None:
        st.write("Prediction probabilities:")
        st.json({str(int(k)): float(v) for k, v in y_prob.items()})

    # SHAP explanation
    try:
        _plot_shap_for_prediction(model, model_features, X_row, int(y_pred))
    except Exception as e:
        st.warning(f"Couldn't render SHAP explanation: {type(e).__name__}: {e}")

st.markdown("---")
st.caption("Tip: Change an input (e.g., add/remove AC or Parking) and re-run to see how the SHAP explanation changes.")
