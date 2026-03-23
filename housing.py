import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import streamlit as st         

df = pd.read_csv("Housing copy.csv")   

model = LinearRegression()
x = df[["area"]]
y = df["price"]
model.fit(x, y)
y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)   

# ---- STREAMLIT UI STARTS HERE ----

st.title("House Price Predictor")                        
st.write("Enter the area of the house to get a predicted price.")  

area_input = st.number_input(                           
    "Area in square feet",
    min_value=100,
    max_value=10000,
    value=1000
)

if st.button("Predict Price"):                           
    prediction = model.predict([[area_input]])[0]        
    st.success(f"Predicted Price: ${prediction:,.2f}")   

st.subheader("Model Performance")                       
st.write(f"Mean Squared Error: {mse:,.2f}")             
st.write(f"Slope: {model.coef_[0]:.2f}")                
st.write(f"Intercept: {model.intercept_:.2f}")           

st.subheader("Regression Plot")                          
fig, ax = plt.subplots()                                 
ax.scatter(x, y, color="blue", label="Original data")   
ax.plot(x, y_pred, color="green", label="Linear Regression")  
ax.set_xlabel("Area in sq feet")                         
ax.set_ylabel("Price in dollars")                        
ax.legend()                                              
ax.grid(True)                                            
st.pyplot(fig)                                           
