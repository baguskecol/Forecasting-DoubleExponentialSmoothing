import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.sidebar.markdown(
    """
    <p style='text-align: center; font-size: 25px; font-family: ink free, sans-serif;'>PRODUKSI KOMODITAS KOPI INDONESIA</p>
    """,
    unsafe_allow_html=True
)
st.sidebar.image(image=("kopi2.jpeg"), use_column_width=True)


# Read the Excel file with specified data types


# membaca data
@st.cache
def load_data():
    data = pd.read_excel('data.xlsx', dtype={'Tahun': 'int64'})
    # data = pd.read_excel('data.xlsx')
    return data
data = load_data()

# menjadikan data menjadi data frame
df = pd.DataFrame(data)

st.write("<h1 style='text-align: center;'>Analisis Time Series Menggunakan Metode Double Exponential Smoothing dalam Memprediksi Produksi Komoditas Kopi di Indonesia</h1>", unsafe_allow_html=True)

with st.expander(
    "Data Selection", expanded=False
):
    st.write(data)
    st.write("Note :")
    st.write("""Data yang digunakan merupakan data sekunder yang di ambil dari sebuah situs 
            pertanian yaitu ppid.go.id https://satudata.pertanian.go.id/datasets.""")
unsafe_allow_html=True
st.write("")

# menghapus data luas areal
data = data.drop(['Luas Areal (ha)','Areal Perkebunan Rakyat', 'Areal Perkebunan Besar Negara', 'Areal Perkebunan Besar Swasta', 
                'Produksi Perkebunan Rakyat', 'Produksi Perkebunan Besar Negara', 'Produksi Perkebunan Besar Swasta'], axis=1)

with st.expander(
    "Data Preprocessing / Cleaning", expanded=False
):
    st.write(data)
    st.write("Note :")
    st.write("""Atribut data yang akan digunakan adalah data Tahun dan jumlah Produksi (ton), 
            dan membuang data yang tidak diperlukan, seperti data Luas Areal (ha) dan data 
            Produksi Perkebunan Rakyat (PR), Produksi Perkebunan Besar Negara (PBN), Produksi 
            Perkebunan Besar Swasta (PBS).""")
st.write("")

# mengambil nilai jumlah produksi
series = data ['Produksi (ton)'].values
with st.expander(
    "Data Transformation", expanded=False
):
    st.write("")
    st.line_chart(df[['Produksi (ton)']])
    st.write("Note :")
    st.write("""Untuk kolom yang digunakan hanya kolom tahun dan produksi (ton) dari tahun 1980 sampai 
            tahun 2022 dan data tersebut diubah menjadi sebuah data time series.""")
st.write("")

st.sidebar.header('Menentukan Nilai Alpha dan Beta :')
# st.sidebar.info('Menentukan Nilai Alpha dan Beta')

# memasukan nilai alpha dan beta secara manual
alpha = st.sidebar.slider('alpha', 0.0, 1.0, 0.5)
beta = st.sidebar.slider('beta', 0.0, 1.0, 0.5)

# menghitung nilai level pertama dan trend pertama
level = [series[0]]
trend = [series[1] - series[0]]

# menghitung nilai level kedua dan trend kedua
level.append(alpha * series[1] + (1 - alpha) * (level[0] + trend[0]))
trend.append(beta * (level[1] - level[0]) + (1 - beta) * trend[0])

# menghitung nilai level ketiga dan trend ketiga
level.append(alpha * series[2] + (1 - alpha) * (level[1] + trend[1]))
trend.append(beta * (level[2] - level[1]) + (1 - beta) * trend[1])

# menghitung nilai level ke-n dan trend ke-n
for i in range(3, len(series)):
    level.append(alpha * series[i] + (1 - alpha) * (level[i - 1] + trend[i - 1]))
    trend.append(beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1])

# masukan data ke dalam data frame
df = pd.DataFrame({'Tahun': data['Tahun'], 'Produksi (ton)': series, 'Level': level, 'Trend': trend})

# menambakhan kolom tahun 2022
df.loc[len(df)] = [2023, np.nan, 0, 0]

# menghitung nilai prediksi
forecast = [level[0] + trend[0]]
for n in range(0, len(series)):
    forecast.append(level[n] + trend[n])
# masukan data ke dalam data frame
df['Forecast'] = forecast

# menghitung nilai MAD
mad = []
for i in range(0, len(series)):
    mad.append(abs(series[i] - forecast[i]))
mean_mad = sum(mad) / len(series)
mad.append(mean_mad)
df['MAD'] = mad

# menghitung nilai MSE
mse = []
for i in range(0, len(series)):
    mse.append((series[i] - forecast[i]) ** 2)
mean_mse = sum(mse) / len(series)
mse.append(mean_mse)
df['MSE'] = mse

# menghitung nilai MAPE
mape = []
for i in range(0, len(series)):
    mape.append(abs(series[i] - forecast[i]) / series[i])
sum_mape = sum(mape) / len(series)*100
mape.append(sum_mape)
df['MAPE'] = mape

# menampilkan data frame
st.subheader("Data Mining")

df['Tahun'] = df['Tahun'].astype('int64')
st.write(df.iloc[0:43])
st.write("")

forecast = st.metric("HASIL PREDIKSI PADA TAHUN 2023 :", forecast[-1], "ton")
mean_mad = st.metric("MAD :", mean_mad)
mean_mse = st.metric("MSE :", mean_mse)
sum_mape = st.metric("MAPE :", sum_mape, "%")

# menggunakan metode grid search untuk mencari nilai alpha dan beta optimal dengan menggunakan Mean Absolute Percentage Error (MAPE)
# menghitung nilai MAPE
def grid_search(series, alpha, beta):
    level = [series[0]]
    trend = [series[1] - series[0]]
    for i in range(1, len(series)):
        level.append(alpha * series[i] + (1 - alpha) * (level[i - 1] + trend[i - 1]))
        trend.append(beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1])
    forecast = [level[0] + trend[0]]
    for i in range(0, len(series)):
        forecast.append(level[i] + trend[i])
    mape = []
    for i in range(0, len(series)):
        mape.append(abs(series[i] - forecast[i]) / series[i])
    sum_mape = sum(mape) / len(series)*100
    return sum_mape
    
    # mencari nilai alpha dan beta optimal
def find_optimal(series):
    alpha = 0
    beta = 0
    min_mape = 100
    for i in range(0, 100):
        for j in range(0, 100):
            mape = grid_search(series, i/100, j/100)
            if mape < min_mape:
                min_mape = mape
                alpha = i/100
                beta = j/100
    return alpha, beta, min_mape

# menampilkan nilai alpha dan beta optimal
alpha, beta, min_mape = find_optimal(series)
with st.expander(
    "Data Evaluasi", expanded=False
):
    st.write("Nilai alpha dan beta optimal adalah :")
    st.write("alpha ", alpha)
    st.write("beta ", beta)
    st.write("MAPE ", min_mape , "%")
    st.write("Note :")
    st.write("""Menggunakan metode grid search untuk mencari nilai alpha dan beta optimal dengan menggunakan 
            Mean Absolute Percentage Error (MAPE).""")

fig = px.line(df, x='Tahun', y=['Produksi (ton)', 'Forecast'], title='Produksi Komoditas Kopi di Indonesia')
st.plotly_chart(fig)

st.subheader('Kesimpulan :')
text = """Berdasarkan hasil analisis dapat disimpulkan bahwa peramalan menggunakan metode Double Exponential 
        Smoothing jika melihat dari tabel kategori berdasarkan nilai MAPE berada dibawah 10% menandakan 
        bahwa hasil peramalan pada jumlah produksi komoditas kopi di Indonesia masuk ke dalam kategori 
        sangat baik untuk digunakan sebagai pemodelan untuk peramalan dimasa yang akan datang. Dari hasil 
        peramalan yang diperoleh menggunakan metode Double Exponential Smoothing menunjukkan adanya kenaikan 
        pada 5 tahun terakhir dari hasil peramalan tahun 2018 sampai 2022 (716094, 761305, 766101, 774615, 
        786210) dan hasil peramalan pada jumlah produksi di tahun 2023 adalah 804137 (ton)."""
st.caption(text.rjust(60))

# menggunakan nilai alpha dan beta optimal untuk peramalan
# menghitung nilai peramalan ke-n
def forecast(series, alpha, beta):
    level = [series[0]]
    trend = [series[1] - series[0]]
    for i in range(1, len(series)):
        level.append(alpha * series[i] + (1 - alpha) * (level[i - 1] + trend[i - 1]))
        trend.append(beta * (level[i] - level[i - 1]) + (1 - beta) * trend[i - 1])
    forecast = [level[0] + trend[0]]
    for i in range(0, len(series)):
        forecast.append(level[i] + trend[i])
    return forecast
    # menghitung nilai peramalan ke-n
forecast = forecast(series, alpha, beta)
df['Forecast'] = forecast

# menghitung nilai MAD
mad = []
for i in range(0, len(series)):
    mad.append(abs(series[i] - forecast[i]))

# rata rata nilai MAD
mean_mad = sum(mad) / len(series)

# tambahkan nilai hasil jumlah MAD ke nilai MAD
mad.append(mean_mad)

# menghitung nilai MSE
mse = []
for i in range(0, len(series)):
    mse.append((series[i] - forecast[i]) ** 2)

# rata rata nilai MSE
mean_mse = sum(mse) / len(series)

# tambahkan nilai hasil jumlah MSE ke nilai MSE
mse.append(mean_mse)

# menghitung nilai MAPE
mape = []
for i in range(0, len(series)):
    mape.append(abs(series[i] - forecast[i]) / series[i])

# rata rata nilai MAPE
sum_mape = sum(mape) / len(series)*100

# tambahkan nilai hasil jumlah MAPE ke nilai MAPE
mape.append(sum_mape)

# masukan nilai MAD, MSE, MAPE ke dalam data frame
df['MAD'] = mad
df['MSE'] = mse
df['MAPE'] = mape

# menambahkan kolom error
df['Error'] = df['Produksi (ton)'] - df['Forecast']

# menyimpan data frame ke dalam file excel
df.to_excel('Hasil Peramalan.xlsx', index=False)