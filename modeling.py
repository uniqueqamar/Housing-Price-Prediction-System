"""Shared, deterministic training and holdout evaluation for the CLI and app."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent / 'HousingData.csv'
NUMERIC = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
CATEGORICAL = ['airconditioning', 'prefarea', 'furnishingstatus']
FEATURES = NUMERIC + CATEGORICAL


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = FEATURES + ['price']
    if df[required].isna().any().any():
        raise ValueError('Missing values in model columns; inspect the dataset before training.')
    for col, allowed in {'airconditioning': {'yes', 'no'}, 'prefarea': {'yes', 'no'},
                         'furnishingstatus': {'furnished', 'semi-furnished', 'unfurnished'}}.items():
        if not set(df[col]).issubset(allowed):
            raise ValueError(f'Unexpected categories in {col}')
    return df


def scores(y, prediction):
    return {'mae_inr': float(mean_absolute_error(y, prediction)),
            'rmse_inr': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction))}


def train_and_evaluate(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], df['price'], test_size=0.2, random_state=42)
    preprocessing = ColumnTransformer([
        ('numeric', 'passthrough', NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='error', drop='first'), CATEGORICAL)])
    model = Pipeline([('preprocessing', preprocessing), ('regression', LinearRegression())])
    model.fit(x_train, y_train)
    baseline = DummyRegressor(strategy='mean').fit(x_train, y_train)
    prediction = model.predict(x_test)
    report = {'dataset_sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
              'random_state': 42, 'test_fraction': 0.2,
              'train_rows': len(x_train), 'test_rows': len(x_test),
              'linear_regression': scores(y_test, prediction),
              'mean_baseline': scores(y_test, baseline.predict(x_test))}
    return model, report, y_test, prediction
