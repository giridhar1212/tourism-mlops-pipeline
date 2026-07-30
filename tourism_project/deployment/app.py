
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Visit with Us — Wellness Package Predictor",
    page_icon="✈️",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading prediction model …")
def load_model():
    
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    return joblib.load(model_path)

model = load_model()
st.title("✈️ Visit with Us — Wellness Tourism Package Predictor")
st.markdown(
    """
    Complete the customer profile below and click **Predict** to find out
    whether this customer is likely to purchase the
    **Wellness Tourism Package**.
    """
)
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business Owner", "Large Business Owner"],
    )
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input(
        "Monthly Income (₹)", min_value=0, max_value=200_000, value=20_000, step=500
    )

with col2:
    st.subheader("✈️ Travel Preferences")
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    number_of_trips = st.number_input(
        "Annual Trips (avg)", min_value=0, max_value=22, value=2
    )
    number_of_person_visiting = st.number_input(
        "Persons Visiting", min_value=1, max_value=10, value=2
    )
    number_of_children_visiting = st.number_input(
        "Children Visiting (< 5 yrs)", min_value=0, max_value=5, value=0
    )
    preferred_property_star = st.selectbox(
        "Preferred Hotel Star Rating", [3, 4, 5]
    )
    passport = st.selectbox(
        "Has Passport?", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    own_car = st.selectbox(
        "Owns a Car?", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

with col3:
    st.subheader("📞 Sales Interaction")
    type_of_contact = st.selectbox(
        "Type of Contact", ["Self Enquiry", "Company Invited"]
    )
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    )
    duration_of_pitch = st.number_input(
        "Duration of Pitch (mins)", min_value=0, max_value=60, value=10
    )
    number_of_followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=10, value=3
    )
    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
    )

st.divider()

if st.button("🔍  Predict", use_container_width=True, type="primary"):

    # Build a single-row DataFrame with the same column names the model saw
    # during training (all 18 feature columns, excluding CustomerID and
    # ProdTaken which were dropped / used as target).
    input_data = pd.DataFrame([{
        "Age":                      age,
        "TypeofContact":            type_of_contact,
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_of_pitch,
        "Occupation":               occupation,
        "Gender":                   gender,
        "NumberOfPersonVisiting":   number_of_person_visiting,
        "NumberOfFollowups":        number_of_followups,
        "ProductPitched":           product_pitched,
        "PreferredPropertyStar":    preferred_property_star,
        "MaritalStatus":            marital_status,
        "NumberOfTrips":            number_of_trips,
        "Passport":                 passport,
        "PitchSatisfactionScore":   pitch_satisfaction_score,
        "OwnCar":                   own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation":              designation,
        "MonthlyIncome":            monthly_income,
    }])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    if prediction == 1:
        st.success(
            f"### ✅  Will Purchase the Wellness Tourism Package!\n"
            f"Predicted purchase probability: **{probability:.1%}**"
        )
    else:
        st.warning(
            f"### ❌  Unlikely to Purchase\n"
            f"Predicted purchase probability: **{probability:.1%}**"
        )

    with st.expander("📊 View input data"):
        st.dataframe(input_data)

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption("Visit with Us | MLOps Wellness Tourism Package Predictor | Powered by XGBoost & Streamlit")

print("\napp.py written successfully.")
