import streamlit as st
import functions as f
# Title Tab
st.set_page_config(page_title="Dashboard Utama 📊")

# Variable
price_gold_dataset = f.getDataset()
# last_data = 




# Header Page
st.markdown("<h1 style='text-align: center;'>Dashboard Harga Emas <br> Tahun 2023</h1>", unsafe_allow_html=True)

# Sidebar Page
st.sidebar.markdown("# Tahun 2023 📊")

st.sidebar.write("Harga Terkini :    990rb")
st.sidebar.write("Rentang Harga :    990rb")
st.sidebar.write("Rata-rata     :    990rb")


# Body Page
