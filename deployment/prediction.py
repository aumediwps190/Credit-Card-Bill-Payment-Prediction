import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model

with open ('model.pkl', 'rb') as file1:
    modelSVM = pickle.load(file1)

with open ('minmaxscale.pkl', 'rb') as file2:
    minMaxScale = pickle.load(file2)

with open ('robustscale.pkl', 'rb') as file3:
    robustScale = pickle.load(file3)        

# Buat formnya

def run():
    with st.form('formCredBillPay'):
        limitBalance = st.number_input('Limit balance', value = 0.0, help = 'Jumlah anggaran individu')
        sex = st.number_input('Jenis kelamin (1: Pria, 2: Wanita)', min_value=1, max_value=2, value=1)
        edLevel = st.number_input('Tingkat pendidikan (1: Pascasarjana, 2: Universitas, 3: SMA, 4: Lainnya, 5 & 6: Tidak diketahui)', min_value=1, max_value=6, value=2)
        maritalStats = st.number_input('Status pernikahan (1: Menikah, 2: Single, 3: Lainnya)', min_value=1, max_value=3, value=2)
        age = st.number_input('Umur', value=25)
        pay0 = st.number_input('Status pembayaran September:', min_value = -2.0, max_value = 9.0, value = 0.0, step = 1.0, help = 'Isi angka dari -2 hingga 9')
        pay2 = st.number_input('Status pembayaran Agustus:', min_value = -2.0, max_value = 9.0, value = 0.0, step = 1.0, help = 'Isi angka dari -2 hingga 9')
        pay3 = st.number_input('Status pembayaran Juli:', min_value = -2.0, max_value = 9.0, value = 0.0, step = 1.0, help = 'Isi angka dari -2 hingga 9')
        pay4 = st.number_input('Status pembayaran Juni:', min_value = -2.0, max_value = 9.0, value = 0.0, step = 1.0, help = 'Isi angka dari -2 hingga 9')
        pay5 = st.number_input('Status pembayaran Mei:', min_value = -2.0, max_value = 9.0, value = 0.0, step = 1.0, help = 'Isi angka dari -2 hingga 9')
        pay6 = st.number_input('Status pembayaran April:', min_value = -2.0, max_value = 9.0, value = 0.0, step = 1.0, help = 'Isi angka dari -2 hingga 9')
        billAmt1 = st.number_input('Tagihan September:', min_value = 0.0, value = 0.0)
        billAmt2 = st.number_input('Tagihan Agustus:', min_value = 0.0, value = 0.0)
        billAmt3 = st.number_input('Tagihan Juli:', min_value = 0.0, value = 0.0)
        billAmt4 = st.number_input('Tagihan Juni:', min_value = 0.0, value = 0.0)
        billAmt5 = st.number_input('Tagihan Mei:', min_value = 0.0, value = 0.0)
        billAmt6 = st.number_input('Tagihan April:', min_value = 0.0, value = 0.0)
        payAmt1 = st.number_input('Jumlah Terbayar September:', min_value = 0.0, value = 0.0)
        payAmt2 = st.number_input('Jumlah Terbayar Agustus:', min_value = 0.0, value = 0.0)
        payAmt3 = st.number_input('Jumlah Terbayar Juli:', min_value = 0.0, value = 0.0)
        payAmt4 = st.number_input('Jumlah Terbayar Juni:', min_value = 0.0, value = 0.0)
        payAmt5 = st.number_input('Jumlah Terbayar Mei:', min_value = 0.0, value = 0.0)
        payAmt6 = st.number_input('Jumlah Terbayar April:', min_value = 0.0, value = 0.0)

        # Submit button
        submitted = st.form_submit_button('Predict')

    newData = {
        'limit_balance': limitBalance,
        'sex' : sex,
        'education_level': edLevel,
        'marital_status' : maritalStats,
        'age' : age,
        'pay_0': pay0,
        'pay_2': pay2,
        'pay_3': pay3,
        'pay_4': pay4,
        'pay_5': pay5,
        'pay_6': pay6,
        'bill_amt_1' : billAmt1,
        'bill_amt_2' : billAmt2,
        'bill_amt_3' : billAmt3,
        'bill_amt_4' : billAmt4,
        'bill_amt_5' : billAmt5,
        'bill_amt_6' : billAmt6,
        'pay_amt_1' : payAmt1,
        'pay_amt_2' : payAmt2,
        'pay_amt_3' : payAmt3,
        'pay_amt_4' : payAmt4,
        'pay_amt_5' : payAmt5,
        'pay_amt_6' : payAmt6
    }    

    newDataDF = pd.DataFrame(newData, index=[0])

    # Jika tombol submit ditekan
    if submitted:
        # Split antara kolom kategorik dan numerik
        newDataNum = newDataDF.drop(['sex', 'education_level', 'marital_status', 'pay_0',
                         'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6'], axis=1)
        
        newDataCat = newDataDF.drop(['limit_balance', 'age', 'bill_amt_1', 'bill_amt_2', 'bill_amt_3',
                         'bill_amt_4', 'bill_amt_5', 'bill_amt_6', 'pay_amt_1', 'pay_amt_2',
                         'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6'], axis=1)
        
        # Pisahkan data numerik antara age dan yang lainnya
        newDataNumNoAge = newDataNum.drop(['age'], axis=1)
        newDataNumAge = newDataNum[['age']]

        # Scaling data baru
        newDataNumNoAgeScled = robustScale.transform(newDataNumNoAge)
        newDataNumAgeScled = minMaxScale.transform(newDataNumAge)

        # Pembuatan DF untuk data baru yang telah discale
        newDataNumNoAgeScledDF = pd.DataFrame(newDataNumNoAgeScled, columns=['limit_balance', 'bill_amt_1', 'bill_amt_2', 'bill_amt_3',
                         'bill_amt_4', 'bill_amt_5', 'bill_amt_6', 'pay_amt_1', 'pay_amt_2',
                         'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6'])
        
        newDataNumAgeScledDF = pd.DataFrame(newDataNumAgeScled, columns=['age'])

        # Penggabungan dua data baru numerik
        newDataNumScl = pd.concat([newDataNumNoAgeScledDF, newDataNumAgeScledDF], axis=1)

        # Penggabungan dataframe data baru kategorik dan numerik
        newDataFinal = pd.concat([newDataNumScl, newDataCat], axis=1)

        # Predict
        hasilPrediksi = modelSVM.predict(newDataFinal)

        # Tulis hasil prediksi
        if hasilPrediksi == 0:
            st.write('## Individu *tidak akan* "default payment" bulan depan')
        elif hasilPrediksi == 1:
            st.write('## Individu *akan* "default payment" bulan depan')    


if __name__ == '__main__':
    run()
