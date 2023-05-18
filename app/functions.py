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
    dataset_path = 'files/preprocessed_dataset.xlsx'
    preprocessed_dataset = pd.read_excel(dataset_path)
    # Set index Tanggal to data
    preprocessed_dataset['Tanggal'] = pd.to_datetime(preprocessed_dataset.Tanggal, dayfirst=True)
    preprocessed_dataset.set_index(preprocessed_dataset.Tanggal, inplace=True)
    # Drop unused columns
    preprocessed_dataset.drop(columns=['Unnamed: 0', 'Tanggal'], inplace=True)
    return preprocessed_dataset

def getPerYearDataset(year):
    df = getDataset()
    this_year_dataset = df.loc[df.index.year == year]
    return this_year_dataset
    
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
def createLinePlot(DataFrame, x_axes, y_axes, title='Plot'):
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
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 30, 'color': '#656EF2'},
        }, 
        xaxis_title='Waktu',
    )
    return fig


def createBarPlot(DataFrame, x_axes, y_axes, title='Plot'):
    # Membuat diagram batang interaktif menggunakan Plotly Express
    fig = px.bar(DataFrame, x=x_axes, y=y_axes)

    # Mengatur judul dan label sumbu
    title_plot = {
        'text': title,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 20, 'color': '#656EF2'},
    }
    fig.update_layout(
        title=title_plot,
        xaxis_title='Waktu',
        yaxis_title='Harga Emas'
    )

    return fig

def plotPerYear(dataset, title='2023'):
    # Convert to Dataframe
    dataset_df = pd.DataFrame(dataset)
    # Show Plot
    x_axes = dataset_df.index
    y_axes = dataset_df['Harga']
    fig_plot = createLinePlot(dataset_df, x_axes, y_axes, title=title)
    st.plotly_chart(fig_plot)

def plotPerMonth(dataset, title):
    # Convert to Dataframe
    dataset_df = pd.DataFrame(dataset)
    # Show Plot
    x_axes = dataset_df.index
    y_axes = dataset_df['Harga']
    fig_plot = createBarPlot(dataset_df, x_axes, y_axes, title=title)
    st.plotly_chart(fig_plot)

def plotForecasetResult(results_data):
    # Convert to Dataframe
    results_data_df = pd.DataFrame(results_data)
    # Show Plot
    x_axes = results_data_df.index
    y_axes = results_data_df[0]
    fig_plot = createLinePlot(results_data_df, x_axes, y_axes, title="Forecaset Results")
    st.plotly_chart(fig_plot)
