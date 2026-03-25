import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv("HousingData.csv")

# ENCODING: convert yes/no and text columns to numbers
# the model only understands numbers, not words like "yes" or "furnished"
le = LabelEncoder()
df["furnishingstatus"] = le.fit_transform(df["furnishingstatus"])  
# furnished=0, semi-furnished=1, unfurnished=2

for col in ["mainroad", "guestroom", "basement", 
            "hotwaterheating", "airconditioning", "prefarea"]:
    df[col] = df[col].map({"yes": 1, "no": 0})
# converts yes -> 1, no -> 0 for all binary columns

# FEATURES: now using all useful columns, not just area
x = df[["area", "bedrooms", "bathrooms", "stories", 
        "parking", "airconditioning", "prefarea", 
        "furnishingstatus"]]
y = df["price"]

model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)  # r2 tells how accurate model is, 1.0 = perfect

# ---- STREAMLIT UI ----

st.title("House Price Predictor")
st.write("Dataset: 545 houses — prices in Indian Rupees")

st.subheader("Enter House Details")

col1, col2 = st.columns(2)   # creates two side-by-side columns in the UI

with col1:
    area = st.number_input("Area (sqft)", 1650, 16200, 5000)
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
    bathrooms = st.slider("Bathrooms", 1, 4, 1)
    stories = st.slider("Stories (floors)", 1, 4, 2)

with col2:
    parking = st.slider("Parking spots", 0, 3, 1)
    aircon = st.selectbox("Air Conditioning", ["No", "Yes"])
    prefarea = st.selectbox("Preferred Area", ["No", "Yes"])
    furnishing = st.selectbox("Furnishing", 
                    ["Furnished", "Semi-Furnished", "Unfurnished"])

# convert UI inputs back to numbers the model understands
aircon_val = 1 if aircon == "Yes" else 0
prefarea_val = 1 if prefarea == "Yes" else 0
furnish_map = {"Furnished": 0, "Semi-Furnished": 1, "Unfurnished": 2}
furnish_val = furnish_map[furnishing]

if st.button("Predict Price"):
    input_data = [[area, bedrooms, bathrooms, stories, 
                   parking, aircon_val, prefarea_val, furnish_val]]
    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Price: ₹{prediction:,.0f}")
    st.info(f"That is ₹{prediction/100000:.2f} Lakhs")  # shows in lakhs too

st.subheader("Model Accuracy")
st.write(f"R² Score: {r2:.2f} out of 1.0")   # how accurate the model is
st.write(f"Mean Squared Error: ₹{mse:,.0f}")

st.subheader("Price vs Area")
fig, ax = plt.subplots()
ax.scatter(df["area"], y, color="blue", alpha=0.4, label="Actual prices")
ax.scatter(df["area"], y_pred, color="green", alpha=0.4, label="Predicted prices")
ax.set_xlabel("Area in sqft")
ax.set_ylabel("Price in Rupees")
ax.legend()
ax.grid(True)
st.pyplot(fig)
