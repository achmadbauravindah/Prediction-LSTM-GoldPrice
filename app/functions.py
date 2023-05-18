import tensorflow as tf
import numpy as np
import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, timedelta


# Get and Build Data from Excel
def getDataset():
    # Get data from directory
    dataset_path = './preprocessed_dataset.xlsx'
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

def normData(values):
    with open('./scaler_model.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler.transform(values)

def denormmData(values):
    with open('./scaler_model.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler.inverse_transform(values)

# Model forecast Last Data in N-After Days
def modelForecast(dataset, n_days):
    last_data_window = np.array(dataset[-500:])
    last_data_window_normed = normData(last_data_window).reshape(1, -1) # reshape for model LSTM input
    forecasted_values = []
    model = tf.keras.models.load_model("./lstm-model-0.01mae.hdf5")
    for n in range(n_days):
        # forecast Values
        forecasted_value = model.predict(last_data_window_normed, verbose=0)
        # Add forecasted Values to List
        forecasted_values.append(forecasted_value)
        # Slice last_data_window_normed to new data with forecasted values
        last_data_window_normed = np.append(last_data_window_normed, forecasted_value)
        last_data_window_normed = last_data_window_normed[1:].reshape(1, -1)
    # Denormalization forecasted Values
    forecasted_values = np.array(forecasted_values).reshape(1, -1)
    forecasted_values_denorm = denormmData(forecasted_values).reshape(-1)
    return forecasted_values_denorm


def mainForecast(dataset, n_days):
    forecasted_values = modelForecast(dataset, n_days)
    last_date = dataset.index[-1]
    future_timestamps = []
    for i in range(n_days):
        if i < n_days:
            future_timestamp = last_date + timedelta(days=i)
            future_timestamps.append(future_timestamp)
        else:
            break
    forecasted_df = pd.DataFrame(data=forecasted_values, index=future_timestamps, columns=['Harga'])
    return forecasted_df



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


def plotForecasetResult(results_data, title):
    # Convert to Dataframe
    results_data_df = pd.DataFrame(results_data)
    # Show Plot
    x_axes = results_data_df.index
    y_axes = results_data_df['Harga']
    fig_plot = createLinePlot(results_data_df, x_axes, y_axes, title=title)
    st.plotly_chart(fig_plot)
