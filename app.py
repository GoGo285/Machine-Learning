import os, json
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, "models", "xgb_model.pkl"))
    prep = joblib.load(os.path.join(BASE, "models", "preprocessor.pkl"))
    with open(os.path.join(BASE, "models", "feature_columns.json")) as f:
        cols = json.load(f)
    with open(os.path.join(BASE, "models", "cat_options.json")) as f:
        opts = json.load(f)
    return model, prep, cols, opts


model, prep, feature_columns, cat_options = load_artifacts()

EDUCATION = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

st.title("Adult Income Predictor")
st.write("Predict whether a person earns more than $50K per year.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 17, 90, 38)
    education = st.selectbox("Education", list(EDUCATION.keys()), index=8)
    hours = st.slider("Hours per week", 1, 99, 40)
    capital_gain = st.number_input("Capital gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital loss", 0, 4356, 0)
with col2:
    workclass = st.selectbox("Workclass", cat_options["workclass"])
    marital = st.selectbox("Marital status", cat_options["marital.status"])
    occupation = st.selectbox("Occupation", cat_options["occupation"])
    relationship = st.selectbox("Relationship", cat_options["relationship"])
    race = st.selectbox("Race", cat_options["race"])
    sex = st.selectbox("Sex", cat_options["sex"])
    country = st.selectbox("Native country", cat_options["native.country"])

if st.button("Predict"):
    row = pd.DataFrame(0, index=[0], columns=feature_columns)
    row["age"] = age
    row["education.num"] = EDUCATION[education]
    row["hours.per.week"] = hours
    row["capital.gain"] = capital_gain
    row["capital.loss"] = capital_loss

    chosen = {
        "workclass": workclass,
        "marital.status": marital,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "native.country": country,
    }
    for col, val in chosen.items():
        row[f"{col}_{val}"] = 1

    X = prep.transform(row)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0, 1]

    if pred == 1:
        st.success(f"Predicted income: **>50K** (probability {proba:.1%})")
    else:
        st.info(f"Predicted income: **<=50K** (probability of >50K: {proba:.1%})")
