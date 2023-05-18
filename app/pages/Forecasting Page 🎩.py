import streamlit as st
import functions as f
from datetime import datetime

##################################### ALL VARIABLE USED #####################################
local_session = {'n_days_forecast': 0, 'max_value': '⏳', 'min_value': '⏳'}
dataset = f.getDataset()


########################################## PAGES ##########################################
# TITLE TAB
st.set_page_config(page_title="Forecasting Harga Emas 🎩")

# HEADER
st.markdown("<h1 style='text-align: center;'>Forecasting Harga Emas 🎩</h1>", unsafe_allow_html=True)


# BODY
# Create column in page
buff, col, buff2 = st.columns([1, 1, 1])
state = col.empty()
input_n_days = state.text_input("Masukan jumlah hari forecasting (maks: 15 hari)", max_chars=2)
if input_n_days:
    state.empty()
    local_session['n_days_forecast'] = input_n_days
    n_days_forecasting = int(local_session['n_days_forecast'])
    if n_days_forecasting >= 15:
        n_days_forecasting = 15
    forecasting_results = f.mainForecast(dataset, n_days=n_days_forecasting)
    f.plotForecasetResult(forecasting_results, 'Hasil Forecasting Selama {} Hari'.format(n_days_forecasting))
    is_up_or_down = ''
    if forecasting_results['Harga'][-1] > forecasting_results['Harga'][0]:
        is_up_or_down = 'Naik'
    elif forecasting_results['Harga'][-1] < forecasting_results['Harga'][0]:
        is_up_or_down = 'Turun'
    else: 
        is_up_or_down = 'Tetap'
    
    max_value = int(forecasting_results['Harga'].max())/1000
    max_value_date = forecasting_results['Harga'].idxmax()
    min_value = int(forecasting_results['Harga'].min())/1000
    min_value_date = forecasting_results['Harga'].idxmin()
    str_max = "{}rb ({})".format(str(max_value), datetime.strftime(max_value_date, "%d-%m-%Y"))
    str_min = "{}rb ({})".format(str(min_value), datetime.strftime(min_value_date, "%d-%m-%Y"))
    local_session['max_value'] = str_max
    local_session['min_value'] = str_min
    st.markdown(
        f'<h3 style="text-align: center;">Selama {n_days_forecasting} hari kedepan Harga Emas akan {is_up_or_down}</h3>', unsafe_allow_html=True)


# SIDEBAR (in here because, there is value to update is max or min price)
st.sidebar.markdown("# Forecasting 🎩")
st.sidebar.write("Harga Tertinggi:")
st.sidebar.write(local_session['max_value'])
st.sidebar.write("Harga Terendah :")
st.sidebar.write(local_session['min_value'])
with st.sidebar:
    st.markdown(
        f'<span style="font-size:12px; font-style:italic;">Terakhir diperbarui pada {datetime.strftime(dataset.index[-1],"%d-%m-%Y")}</span>', unsafe_allow_html=True)

# FOOTER
st.markdown("<p style='text-align: center; font-style:italic;'>*For the best experience please turn to light theme and landscape mode (mobile)*</p>", unsafe_allow_html=True)
