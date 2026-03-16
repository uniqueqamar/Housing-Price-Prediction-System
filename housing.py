import numpy as np
import pandas as pd  
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

df = pd.read_csv("/Users/Fatima/python_project/Housing copy.csv")

model = LinearRegression()
x=df[["area"]]
y=df["price"]
model.fit(x,y)


y_pred=model.predict(x)
new_area =[[1000]]
new_price=model.predict(new_area)[0]
print(f"Predicted Price for 1000sq feet is : ${new_price:.2f}")

mse = mean_squared_error(y, y_pred)
print("Mean squared error:", mse)

print("slope:", model.coef_[0])
print("Model intercept: ",model.intercept_)


plt.scatter(x,y,color="blue",label= "Original data")
plt.plot(x,y_pred,color="green",label="Linear Regression")
plt.xlabel("Area in sq feet")
plt.ylabel("price in dollars ")
plt.legend()
plt.grid(True)
plt.savefig("housing_plot.png")








