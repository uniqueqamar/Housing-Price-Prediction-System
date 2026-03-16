House Price Prediction using Linear Regression
This project implements a Machine Learning regression model to predict house prices based on property area using Linear Regression. The model is trained on housing data and visualizes the relationship between house area and price.


The goal of this project is to demonstrate the end-to-end machine learning workflow, including data loading, model training, prediction, evaluation, and visualization.

Project Overview
Housing prices often depend on various factors such as location, size, and amenities. In this project, we focus on area (square feet) as the primary feature and train a Linear Regression model to estimate house prices.

The model learns the relationship between:

Independent variable: Area (square feet)

Dependent variable: Price (dollars)

After training, the model can predict house prices for new areas.

Example prediction in the code:

Area: 1000 sq ft

Model outputs predicted price.

Technologies Used
Python

NumPy – numerical computations

Pandas – data manipulation

Scikit-learn – machine learning model

Matplotlib – data visualization

Machine Learning Model
This project uses Linear Regression, a supervised learning algorithm used for predicting continuous values.


Project Workflow


1. Data Loading
The housing dataset is loaded using Pandas.


2. Feature Selection
Input feature: area

Target variable: price


3. Model Training
The Linear Regression model is trained using Scikit-learn.


4. Prediction
The trained model predicts house prices for:

Existing data

New input values (example: 1000 sq ft)


5. Model Evaluation
Model performance is evaluated using:

Mean Squared Error (MSE)

Lower MSE indicates better prediction performance.


6. Visualization
A scatter plot is created showing:

Original housing data

Linear regression prediction line

The plot is saved as:

housing_plot.png
Example Output
Example output from the program:

Predicted Price for 1000 sq feet is : $XXXX
Mean squared error: XXXX
Slope: XXXX
Model intercept: XXXX
Visualization
The visualization shows:

Blue dots → original housing data

Green line → regression prediction line

This helps visualize the relationship between area and house price.

Project Structure
House-Price-Prediction
│
├── Housing copy.csv
├── housing.py
├── housing_plot.png
└── README.md
How to Run the Project
1. Clone the repository
git clone https://github.com/uniqueqamar/Housing-Price-Prediction-System.git

2. Install required libraries
pip install numpy pandas scikit-learn matplotlib
3. Run the script
python house_price_prediction.py
The program will:

-Train the model

-Predict housing price

-Display evaluation metrics

-Save the visualization plot

Future Improvements
Possible improvements to this project include:

Adding multiple features (bedrooms, location, age of house)

Using multiple linear regression

Applying advanced models such as Random Forest or Gradient Boosting

Deploying the model as a web application

Author
Qamareen Fatima
B.Tech – Electronics and Communication Engineering
Birla Institute of Technology, Mesra



