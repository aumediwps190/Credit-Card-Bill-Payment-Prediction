import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi layout webpage

st.set_page_config(
    # Judul page
    page_title = 'Credit Card Bill Payment - 2005',
    #layout = 'wide',
    #initial_sidebar_state = 'expanded'
)

# Set up the webpage

def run():

    # Title
    st.title('2005 Credit Card Bill Payment Prediction')

    # Sub-header
    st.subheader('Prediksi Pembayaran Tagihan Berdasarkan Data')

    # Deskripsi
    st.write('Webpage ini dibuat untuk menganalisa dataset tagihan kartu kredit di tahun 2005. \
             Dataset berisi sejumlah profil individu, anggaran kartu kredit mereka, informasi tagihan, \
             jumlah terbayar, serta informasi mengenai apakah pembayaran mereka ditunda atau tidak. \
             Tagihan diterbitkan tiap bulan dari bulan April hingga September. Dengan mensubmit data profil \
             sebuah individu, user dapat menentukan apakah individu tersebut akan "default payment" bulan \
             depan atau tidak.')
    
    st.write('Berikut adalah dataset yang bersangkutan:')
    
    # Load Dataframe
    df = pd.read_csv('P1G5_Set_1_aumedi_wibisana.csv')
    st.dataframe(df)
    
    ## VISUALISASI ##
    st.write('Di bawah ini adalah sejumlah gambaran umum terkait dataset')
    st.write('###### *Jumlah masing-masing kelamin (1: pria, 2: wanita) :*')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(data = df, x='sex')
    st.pyplot(fig)

    st.write('###### *Jumlah status pernikahan (1: menikah, 2: single, 3: lainnya) :*')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(data=df, x='marital_status')
    st.pyplot(fig)

    st.write('###### *Jumlah tingkat pendidikan (1: pascasarjana, 2: universitas, 3: SMA, \
             4: lainnya, 5 & 6: unknown) :*')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(data=df, x='education_level')
    st.pyplot(fig)

    st.write('###### *Distribusi usia :*')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data=df, x='age')
    st.pyplot(fig)

    st.write('###### *Distribusi anggaran setiap individu (dalam USD) :*')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data=df, x='limit_balance')
    st.pyplot(fig) 

    st.write('###### *Jumlah entry yang akan "default payment" bulan depan dan tidak (0: tidak, 1: ya) :*')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(data=df, x='default_payment_next_month')
    st.pyplot(fig)

    st.write('Input data untuk prediksi dapat dilakukan di page Prediction.')


if __name__ == '__main__':
    run()  

