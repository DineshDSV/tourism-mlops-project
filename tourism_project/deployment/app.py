import streamlit as st
import pandas as pd
import joblib
import json
from huggingface_hub import hf_hub_download

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered"
)

# ── Load model and metadata ──────────────────────────────────────────
HF_MODEL_REPO = "DineshDSV/tourism-wellness-model"  # ← update

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib")
    meta_path  = hf_hub_download(repo_id=HF_MODEL_REPO, filename="model_metadata.json")
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta

model, metadata = load_model()

# ── UI ───────────────────────────────────────────────────────────────
st.title("✈️ Wellness Tourism Package Predictor")
st.markdown(
    """
    Predict whether a customer will purchase the **Wellness Tourism Package**
    based on their profile and interaction data.
    """
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Profile")
    age                    = st.slider("Age", 18, 75, 35)
    gender                 = st.selectbox("Gender", ["Male", "Female"])
    marital_status         = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation             = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation            = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    city_tier              = st.selectbox("City Tier", [1, 2, 3])
    monthly_income         = st.number_input("Monthly Income (₹)", min_value=5000, max_value=100000, value=25000, step=1000)
    passport               = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    own_car                = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.subheader("Interaction Details")
    type_of_contact        = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product_pitched        = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    duration_of_pitch      = st.slider("Duration of Pitch (mins)", 5, 60, 20)
    pitch_satisfaction     = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    num_followups          = st.slider("Number of Follow-ups", 1, 6, 3)
    num_person_visiting    = st.slider("Number of Persons Visiting", 1, 5, 2)
    num_children_visiting  = st.slider("Number of Children Visiting", 0, 3, 0)
    preferred_star         = st.selectbox("Preferred Property Star", [3, 4, 5])
    num_trips              = st.slider("Number of Trips per Year", 1, 10, 2)

st.divider()
predict_btn = st.button("🔍 Predict Purchase Likelihood", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame([{
        "TypeofContact":           type_of_contact,
        "Occupation":              occupation,
        "Gender":                  gender,
        "ProductPitched":          product_pitched,
        "MaritalStatus":           marital_status,
        "Designation":             designation,
        "Age":                     age,
        "CityTier":                city_tier,
        "DurationOfPitch":         duration_of_pitch,
        "NumberOfPersonVisiting":  num_person_visiting,
        "NumberOfFollowups":       num_followups,
        "PreferredPropertyStar":   preferred_star,
        "NumberOfTrips":           num_trips,
        "Passport":                passport,
        "PitchSatisfactionScore":  pitch_satisfaction,
        "OwnCar":                  own_car,
        "NumberOfChildrenVisiting":num_children_visiting,
        "MonthlyIncome":           monthly_income
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ **Likely to Purchase** the Wellness Tourism Package")
    else:
        st.warning(f"❌ **Unlikely to Purchase** the Wellness Tourism Package")

    st.metric("Purchase Probability", f"{probability * 100:.1f}%")
    st.progress(float(probability))

    st.caption(f"Model: {metadata['model_name']} | ROC-AUC: {metadata['metrics']['roc_auc']}")

st.divider()
st.caption("Built with ❤️ using Streamlit · MLOps Project – Visit with Us")
