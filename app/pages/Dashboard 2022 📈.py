import streamlit as st
import functions as f
from datetime import datetime

##################################### ALL VARIBLE USED #####################################
# Get All Dataset
df_all_year = f.getDataset()
# Get Dataset in 2022
df_2022 = f.getPerYearDataset(2022)
# Get Dataset Per Month in 2022
df_2022_monthly_average = df_2022.resample('M').mean()
# Change Index Data to Month
df_2022_monthly_average.index = df_2022_monthly_average.index.strftime('%B')
# Get Nowdays Price
harga_terkini = str(int((df_2022.values[-1])/1000))
# Get Minimum Price
harga_min = str(int(df_2022.min()/1000))
# Get Maximum Price
harga_max = str(int(df_2022.max()/1000))
# Get Average Value in year
harga_rata2 = str(int(df_2022.mean()/1000))


########################################## PAGES ##########################################
# TITLE TAB
st.set_page_config(page_title="Dashboard 2022 📈")

# HEADER
st.markdown("<h1 style='text-align: center;'>Dashboard Harga Emas</h1>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("# Tahun 2022 📈")
st.sidebar.write("Harga Terkini :    {}rb".format(harga_terkini))
st.sidebar.write("Rentang Harga :    {}rb-{}rb".format(harga_min, harga_max))
st.sidebar.write("Rata-rata     :    {}rb".format(harga_rata2))
with st.sidebar:
    st.markdown(
        f'<span style="font-size:12px; font-style:italic;">Terakhir diperbarui pada {datetime.strftime(df_all_year.index[-1],"%d-%m-%Y")}</span>', unsafe_allow_html=True)


# BODY
f.plotPerYear(df_2022, 'Tahun 2022')
f.plotPerMonth(df_2022_monthly_average, 'Rata-rata Bulanan')

# FOOTER
st.markdown("<p style='text-align: center; font-style:italic;'>*For the best experience please turn to light theme and landscape mode (mobile)*</p>", unsafe_allow_html=True)
