# Pilihan antara page eda atau prediction

import streamlit as st
import eda
import prediction

# Variabel untuk menyimpan apakah tombol 'submitted' sudah terpilih atau belum
page = st.sidebar.selectbox('Pilih halaman:', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()    

