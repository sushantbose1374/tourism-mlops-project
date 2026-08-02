
"""
=========================================================
Tourism Package Purchase Prediction
Streamlit Web Application
=========================================================
"""

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Tourism Package Purchase Prediction")

st.markdown(
"""
Predict whether a customer is likely to purchase
the Wellness Tourism Package.
"""
)

# -------------------------------------------------------
# Load Model Bundle
# -------------------------------------------------------

#_________________________________________________________
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

bundle = joblib.load(MODEL_PATH)

#_________________________________________________________

model = bundle["model"]

label_encoders = bundle["label_encoders"]

feature_columns = bundle["feature_columns"]

# -------------------------------------------------------
# User Inputs
# -------------------------------------------------------

customer_id = st.number_input(
    "Customer ID",
    value=100001
)

age = st.number_input(
    "Age",
    value=35
)

type_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry","Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1,2,3]
)

duration = st.number_input(
    "Duration of Pitch",
    value=10.0
)

occupation = st.selectbox(
    "Occupation",
    [
        "Salaried",
        "Free Lancer",
        "Small Business",
        "Large Business"
    ]
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female",
        "Fe Male"
    ]
)

persons = st.number_input(
    "Number of Persons Visiting",
    value=2
)

followups = st.number_input(
    "Number of Follow Ups",
    value=2.0
)

product = st.selectbox(
    "Product Pitched",
    [
        "Basic",
        "Deluxe",
        "Standard",
        "Super Deluxe",
        "King"
    ]
)

star = st.selectbox(
    "Preferred Property Star",
    [3,4,5]
)

marital = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced",
        "Unmarried"
    ]
)

trips = st.number_input(
    "Number of Trips",
    value=2.0
)

passport = st.selectbox(
    "Passport",
    [0,1]
)

pitch = st.slider(
    "Pitch Satisfaction",
    1,
    5,
    3
)

owncar = st.selectbox(
    "Own Car",
    [0,1]
)

children = st.number_input(
    "Children Visiting",
    value=0.0
)

designation = st.selectbox(
    "Designation",
    [
        "Manager",
        "Executive",
        "Senior Manager",
        "AVP",
        "VP"
    ]
)

income = st.number_input(
    "Monthly Income",
    value=25000.0
)
    # -------------------------------------------------------
# Prediction
# -------------------------------------------------------

input_data = pd.DataFrame([{

    "CustomerID": customer_id,
    "Age": age,
    "TypeofContact": type_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income

}])

# -------------------------------------------------------
# Encode categorical columns
# -------------------------------------------------------

categorical_columns = [

    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"

]

for column in categorical_columns:

    encoder = label_encoders[column]

    input_data[column] = encoder.transform(
        input_data[column]
    )

# Arrange columns in the same order as training

input_data = input_data[feature_columns]

# -------------------------------------------------------
# Predict
# -------------------------------------------------------

if st.button("Predict Purchase"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"✅ Customer is LIKELY to purchase the package.\n\nProbability : {probability:.2%}"
        )

    else:

        st.error(
            f"❌ Customer is UNLIKELY to purchase the package.\n\nProbability : {probability:.2%}"
        )

    st.markdown("---")

    st.write("### Input Summary")

    st.dataframe(input_data)
