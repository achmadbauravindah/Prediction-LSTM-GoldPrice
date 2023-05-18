import tensorflow as tf
import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle


# Get and Build Data from Excel
def getDataset():
    # Get data from directory
    data_path = 'files/historical-gold-data.xlsx'
    data = pd.read_excel(data_path)
    # Set index dates to data
    data["Date"] = pd.to_datetime(data.Date, dayfirst=True)
    data.set_index("Date", inplace=True)
    # Add Day Date with NaN Date Values
    data_new = data.reindex(pd.date_range('1985-01-01', '2023-03-14'))  # ! Change Periodically
    # Impute Missing Values
    price_data = data_new['Price']
    price_data = price_data.fillna(method='ffill')  # Interpolate NaN data (based on before and after value)
    price_data[0] = price_data[1]  # First data can't to interpolate, so this is to rise it
    # Convert USD to IDR
    price_data_rupiah = (price_data/28.3495)*14687
    
    return price_data_rupiah

def getLastData(dataset):
    # Data Normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset_arr = np.array(dataset)
    dataset_norm = scaler.fit_transform(dataset_arr.reshape(-1, 1))
    # Get Last Data (1 Row 500 Window/Column)
    last_data = dataset_norm[-500:].reshape(1, -1)  # Reshape for model predict
    # Save Scaler to Pickle File
    with open('files/scaler_model.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return last_data

def denormData(data):
    with open('files/scaler_model.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler.inverse_transform(data.reshape(-1, 1))

# Model Predict Last Data in N-After Days
def predictAfterNDays(n_days, dataset):
    X_before_days = getLastData(dataset)  # 1 Row 500 Window
    predicted_values = []
    model = tf.keras.models.load_model("files/lstm-model-0.01mae.hdf5")
    for n in range(n_days):
        # Predict Values
        predicted_value = model.predict(X_before_days, verbose=0)
        # Add Predicted Values to List
        predicted_values.append(predicted_value)
        # Slice X_before_days to new data with predicted values
        X_before_days = np.append(X_before_days, predicted_value)
        X_before_days = X_before_days[1:].reshape(1, -1)
    # Denormalization Predicted Values
    predicted_values = np.array(predicted_values)
    predicted_values_denorm = denormData(predicted_values)
    return predicted_values_denorm

# Create Plot with Plot Express
def createPlotExpress(DataFrame, x_axes, y_axes, title='Plot'):

    fig = px.line(DataFrame, x=x_axes, y=y_axes, title=title)
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig

    

def plotDataset(dataset):
    # Convert to Dataframe
    dataset_df = pd.DataFrame(dataset)
    # Show Plot
    x_axes = dataset_df.index
    y_axes = dataset_df['Price']
    fig_plot = createPlotExpress(dataset_df, x_axes, y_axes, title="Dataset")
    st.plotly_chart(fig_plot)

def plotForecasetResult(results_data):
    # Convert to Dataframe
    results_data_df = pd.DataFrame(results_data)
    # Show Plot
    x_axes = results_data_df.index
    y_axes = results_data_df[0]
    fig_plot = createPlotExpress(results_data_df, x_axes, y_axes, title="Forecaset Results")
    st.plotly_chart(fig_plot)
