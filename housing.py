"""Interactive demonstration of the model evaluated by evaluate.py."""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from modeling import FEATURES, load_data, train_and_evaluate

st.set_page_config(page_title='House Price Predictor', layout='centered')

@st.cache_resource
def training():
    df = load_data()
    return df, train_and_evaluate(df)

df, (model, report, actual, predicted) = training()
st.title('House Price Predictor')
st.write(f'An educational linear regression demo using {len(df)} housing records.')
st.info('Prices are displayed in INR as in the original project. Dataset provenance, location, date and area units are not independently verified. This is not a property valuation tool.')
st.subheader('Enter house details')
with st.form('house_details'):
    left, right = st.columns(2)
    with left:
        area = st.number_input('Area (dataset units)', int(df.area.min()), int(df.area.max()), 5000)
        bedrooms = st.slider('Bedrooms', 1, 6, 3)
        bathrooms = st.slider('Bathrooms', 1, 4, 1)
        stories = st.slider('Stories (floors)', 1, 4, 2)
    with right:
        parking = st.slider('Parking spots', 0, 3, 1)
        aircon = st.selectbox('Air conditioning', ['no', 'yes'])
        prefarea = st.selectbox('Preferred area (dataset flag)', ['no', 'yes'])
        furnishing = st.selectbox('Furnishing', ['furnished', 'semi-furnished', 'unfurnished'])
    submitted = st.form_submit_button('Predict price')
if submitted:
    row = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking, aircon, prefarea, furnishing]], columns=FEATURES)
    value = float(model.predict(row)[0])
    if value < 0:
        st.warning('The model returned a negative price, which is invalid for a house. This input is outside its useful behavior.')
    else:
        st.success(f'Estimated price: ₹{value:,.0f} (₹{value / 100000:.2f} lakh)')
    st.caption('A point estimate only; combinations of individually valid inputs may still be unlike the training houses.')
st.subheader('Unseen-data evaluation')
st.caption(f"Fixed 80/20 split, seed 42: {report['train_rows']} training houses and {report['test_rows']} holdout houses. The app uses this same training-only model.")
rows = []
for label, key in [('Linear regression', 'linear_regression'), ('Training-mean baseline', 'mean_baseline')]:
    metrics = report[key]
    rows.append({'Model': label, 'MAE (INR)': round(metrics['mae_inr']),
                 'RMSE (INR)': round(metrics['rmse_inr']), 'R²': round(metrics['r2'], 3)})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption('MAE is average absolute error; RMSE penalizes larger errors. Lower is better. R² compares against the holdout mean, is not percentage accuracy, and can be negative.')
st.subheader('Holdout predictions')
fig, ax = plt.subplots()
ax.scatter(actual / 100000, predicted / 100000, alpha=0.6)
bounds = [min(actual.min(), predicted.min()) / 100000, max(actual.max(), predicted.max()) / 100000]
ax.plot(bounds, bounds, '--', color='gray')
ax.set(xlabel='Actual price (INR lakh)', ylabel='Predicted price (INR lakh)')
st.pyplot(fig)
plt.close(fig)
