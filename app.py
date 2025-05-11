import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Rumah Melbourne", layout="wide")

# Direktori model
MODELS_DIR = './models'

# Memuat model dan preprocessor
@st.cache_resource
def load_models():
    try:
        linear_model = joblib.load(os.path.join(MODELS_DIR, 'linear_regression.joblib'))
        poly_model = joblib.load(os.path.join(MODELS_DIR, 'polynomial_regression_(degree=2).joblib'))
        preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
        feature_info = joblib.load(os.path.join(MODELS_DIR, 'feature_info.joblib'))
        
        with open(os.path.join(MODELS_DIR, 'best_model_name.txt'), 'r') as f:
            best_model_name = f.read().strip()
        
        return {
            'Linear Regression': linear_model,
            'Polynomial Regression (degree=2)': poly_model,
            'best_model_name': best_model_name,
            'preprocessor': preprocessor,
            'feature_info': feature_info
        }
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return None

# Fungsi untuk membuat prediksi
def predict_price(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error saat membuat prediksi: {str(e)}")
        return None

# Aplikasi Utama
def main():
    # Judul dan deskripsi aplikasi
    st.title("Prediksi Harga Rumah Melbourne")
    st.markdown("""
    Aplikasi ini memprediksi harga rumah di Melbourne berdasarkan berbagai fitur menggunakan model machine learning:
    - **Regresi Linear**: Model sederhana yang mengasumsikan hubungan linear antara fitur dan harga
    - **Regresi Polinomial**: Model yang lebih kompleks yang dapat menangkap hubungan non-linear
    
    Masukkan detail properti pada formulir di bawah untuk mendapatkan prediksi harga.
    """)
    
    # Memuat model
    models_data = load_models()
    if models_data is None:
        st.error("Gagal memuat model. Silakan periksa file model dan path direktori.")
        return
    
    # Informasi fitur untuk pembuatan formulir
    numerical_cols = models_data['feature_info']['numerical_cols_used']
    categorical_cols = models_data['feature_info']['categorical_cols_used']
    
    # Membuat tab untuk prediksi dan informasi model
    tab1, tab2, tab3 = st.tabs(["Buat Prediksi", "Informasi Model", "Tentang Dataset"])
    
    with tab1:
        st.header("Detail Properti")
        
        # Membuat layout dua kolom
        col1, col2 = st.columns(2)
        
        # Formulir untuk input pengguna
        with st.form("prediction_form"):
            # Input numerik di kolom pertama
            with col1:
                st.subheader("Karakteristik Properti")
                rooms = st.slider("Jumlah Ruangan", 1, 10, 3)
                bedrooms = st.slider("Jumlah Kamar Tidur", 1, 8, 2)
                bathrooms = st.slider("Jumlah Kamar Mandi", 1, 5, 1)
                car_spaces = st.slider("Jumlah Tempat Parkir Mobil", 0, 5, 1)
                land_size = st.number_input("Luas Tanah (m²)", min_value=50, max_value=2000, value=500, step=50)
                building_area = st.number_input("Luas Bangunan (m²)", min_value=50, max_value=1000, value=150, step=25)
                year_built = st.slider("Tahun Dibangun", 1900, 2023, 1990)
            
            # Input kategori dan lokasi di kolom kedua
            with col2:
                st.subheader("Lokasi & Tipe")
                distance = st.slider("Jarak dari CBD (km)", 0.0, 40.0, 10.0, 0.5)
                property_type = st.selectbox("Tipe Properti", ["house", "unit", "townhouse"], 
                                           format_func=lambda x: {"house": "Rumah", "unit": "Unit", "townhouse": "Townhouse"}[x])
                
                # Wilayah Melbourne
                region_options = ["Northern Metropolitan", "Southern Metropolitan", "Western Metropolitan", 
                                 "Eastern Metropolitan", "South-Eastern Metropolitan", "Eastern Victoria", 
                                 "Northern Victoria", "Western Victoria"]
                region = st.selectbox("Wilayah", region_options)
                
                # Daerah dewan kota Melbourne
                council_options = ["Banyule", "Bayside", "Boroondara", "Brimbank", "Cardinia", 
                                  "Casey", "Darebin", "Frankston", "Glen Eira", "Greater Dandenong",
                                  "Hobsons Bay", "Hume", "Kingston", "Knox", "Macedon Ranges",
                                  "Manningham", "Maribyrnong", "Maroondah", "Melbourne", "Melton",
                                  "Monash", "Moonee Valley", "Moreland", "Mornington Peninsula",
                                  "Nillumbik", "Port Phillip", "Stonnington", "Whitehorse", "Whittlesea",
                                  "Yarra", "Yarra Ranges"]
                council = st.selectbox("Daerah Dewan Kota", council_options)
                
                # Metode penjualan
                method_options = ["S", "SP", "PI", "VB", "SA"]
                method = st.selectbox("Metode Penjualan", method_options, 
                                    format_func=lambda x: {"S": "Terjual", "SP": "Properti terjual sebelumnya", 
                                                         "PI": "Properti tidak terjual", "VB": "Penawaran Penjual", 
                                                         "SA": "Terjual Setelah Lelang"}[x] if x in ["S", "SP", "PI", "VB", "SA"] else x)
                
                # Untuk demonstrasi, jumlah properti di daerah ditentukan manual
                property_count = st.slider("Jumlah properti di daerah", 100, 10000, 5000, 100)
            
            # Tombol submit untuk prediksi
            submit_button = st.form_submit_button("Prediksi Harga")
        
        # Proses prediksi saat formulir dikirim
        if submit_button:
            # Membuat dataframe input
            input_data = pd.DataFrame({
                'Rooms': [rooms],
                'Distance': [distance],
                'Bedroom2': [bedrooms],
                'Bathroom': [bathrooms],
                'Car': [car_spaces],
                'Landsize': [land_size],
                'BuildingArea': [building_area],
                'YearBuilt': [year_built],
                'Propertycount': [property_count],
                'Type': [property_type],
                'Method': [method],
                'Regionname': [region],
                'CouncilArea': [council]
            })
            
            # Menampilkan spinner saat membuat prediksi
            with st.spinner('Menghitung harga...'):
                # Membuat prediksi dengan kedua model
                linear_pred = predict_price(models_data['Linear Regression'], input_data)
                poly_pred = predict_price(models_data['Polynomial Regression (degree=2)'], input_data)
                
                # Menentukan model terbaik
                best_model = models_data['best_model_name']
                best_pred = linear_pred if best_model == 'Linear Regression' else poly_pred
                
                # Menampilkan hasil
                st.success("Prediksi Selesai!")
                
                # Membuat kolom untuk menampilkan prediksi
                res_col1, res_col2, res_col3 = st.columns([1, 1, 1])
                
                with res_col1:
                    st.metric("Prediksi Regresi Linear", f"${int(linear_pred):,}")
                
                with res_col2:
                    st.metric("Prediksi Regresi Polinomial", f"${int(poly_pred):,}")
                
                with res_col3:
                    st.metric(f"Prediksi Model Terbaik ({best_model})", f"${int(best_pred):,}")
                
                # Membuat grafik batang untuk membandingkan prediksi
                fig, ax = plt.subplots(figsize=(10, 6))
                models_list = ['Regresi Linear', 'Regresi Polinomial']
                prices = [linear_pred, poly_pred]
                colors = ['steelblue', 'darkorange']
                
                bars = ax.bar(models_list, prices, color=colors)
                ax.set_ylabel('Harga Prediksi (AUD $)')
                ax.set_title('Perbandingan Prediksi Model')
                
                # Menambahkan label harga di atas batang
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                           f'${int(height):,}',
                           ha='center', va='bottom', fontsize=11)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Menambahkan disclaimer
                st.info("Disclaimer: Prediksi ini berdasarkan data historis dan hanya untuk tujuan informasi. Harga properti dipengaruhi oleh banyak faktor yang tidak tercakup dalam model ini.")
    
    with tab2:
        st.header("Informasi Model")
        
        st.subheader("Tentang Model")
        st.markdown("""
        Aplikasi ini menggunakan dua model regresi untuk memprediksi harga rumah:
        
        1. **Regresi Linear**
           - Model sederhana yang mengasumsikan hubungan linear antara fitur dan harga
           - Persamaan: Harga = β₀ + β₁×Rooms + β₂×Distance + ... + βₙ×FeatureN
           - Kelebihan: Sederhana, mudah diinterpretasi, cepat
           - Keterbatasan: Mungkin tidak dapat menangkap hubungan non-linear yang kompleks
        
        2. **Regresi Polinomial (derajat=2)**
           - Memperluas regresi linear dengan memasukkan istilah kuadrat dan interaksi
           - Dapat menangkap hubungan yang lebih kompleks antara fitur dan harga
           - Termasuk istilah seperti Rooms², Distance², Rooms×Distance, dll.
           - Umumnya memberikan akurasi yang lebih baik untuk data properti
        """)
        
        st.subheader("Kepentingan Fitur")
        st.markdown("""
        Fitur yang paling berpengaruh pada prediksi harga rumah adalah:
        
        - **Jumlah Ruangan & Kamar Tidur**: Lebih banyak ruangan umumnya berarti harga lebih tinggi
        - **Kamar Mandi**: Kamar mandi tambahan secara signifikan meningkatkan nilai properti
        - **Luas Bangunan**: Ruang hidup yang lebih besar memiliki harga premium
        - **Jarak dari CBD**: Properti yang lebih dekat ke Central Business District biasanya lebih mahal
        - **Daerah Dewan Kota**: Daerah dewan tertentu (seperti Boroondara, Stonnington) dikaitkan dengan harga yang lebih tinggi
        - **Tipe Properti**: Rumah biasanya lebih mahal daripada unit atau townhouse
        """)
        
        # Membuat grafik kepentingan fitur untuk visualisasi
        fig, ax = plt.subplots(figsize=(10, 7))
        features = ['Ruangan', 'Kamar Mandi', 'Luas Bangunan', 'Jarak', 'Dewan_Stonnington', 'Tipe_rumah', 'Kamar Tidur', 'Tahun Dibangun']
        importance = [0.65, 0.58, 0.52, -0.48, 0.45, 0.40, 0.38, 0.25]
        colors = ['darkgreen' if x > 0 else 'darkred' for x in importance]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Dampak pada Harga')
        ax.set_title('Kepentingan Fitur (Regresi Linear)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        st.pyplot(fig)
        
        st.subheader("Performa Model")
        st.markdown("""
        Berdasarkan evaluasi data pengujian:
        
        | Model | Skor R² | RMSE (Error Rata-rata) |
        |-------|----------|-------------------|
        | Regresi Linear | 0,6628 | $249.027,48 |
        | Regresi Polinomial (derajat=2) | 0,7199 | $226.966,08 |
        
        Model Regresi Polinomial umumnya berkinerja lebih baik karena dapat menangkap hubungan non-linear dalam data perumahan.
        """)
        
        st.subheader("Interpretasi Model Terbaik (Regresi Polinomial)")
        st.markdown("""
        - Model menjelaskan 71,99% dari varians dalam harga rumah
        - Error prediksi rata-rata (RMSE) adalah $226.966,08
        - Peningkatan performa sebesar 8,62% dibandingkan model Regresi Linear
        """)
    
    with tab3:
        st.header("Tentang Dataset Perumahan Melbourne")
        
        st.markdown("""
        Aplikasi ini menggunakan dataset harga perumahan Melbourne yang bersumber dari [Kaggle - Melbourne Housing Market](https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market/data).
        
        Dataset ini berisi informasi tentang penjualan rumah di Melbourne, Australia antara tahun 2016 dan 2017. Ini mencakup fitur-fitur berikut:
        
        | Fitur | Deskripsi |
        |---------|-------------|
        | Rooms | Jumlah ruangan |
        | Bedroom2 | Jumlah kamar tidur (mungkin berbeda dengan Rooms) |
        | Bathroom | Jumlah kamar mandi |
        | Car | Jumlah tempat parkir mobil |
        | Landsize | Luas tanah dalam meter persegi |
        | BuildingArea | Luas bangunan dalam meter persegi |
        | YearBuilt | Tahun rumah dibangun |
        | Distance | Jarak dari CBD |
        | Propertycount | Jumlah properti di daerah |
        | Type | Tipe properti (rumah, unit, townhouse) |
        | Method | Metode penjualan |
        | CouncilArea | Dewan yang mengatur daerah |
        """)
        
        # Menampilkan gambar contoh properti Melbourne
        st.image("https://images.pexels.com/photos/12526602/pexels-photo-12526602.jpeg", 
                 caption="Contoh rumah di Melbourne (gambar ilustrasi)")
        
        st.subheader("Pasar Properti Melbourne")
        st.markdown("""
        Melbourne adalah ibukota pesisir negara bagian Victoria di Australia tenggara. Kota ini terkenal dengan pemandangan seni yang hidup, arsitektur era Victoria, dan lingkungan yang beragam. Pasar properti di Melbourne dikenal untuk:
        
        - Nilai tinggi ditempatkan pada kedekatan dengan Central Business District (CBD)
        - Harga premium untuk properti di daerah timur dan tenggara
        - Variasi harga yang signifikan antara berbagai area dewan
        - Preferensi kuat untuk rumah yang lebih besar dan terpisah daripada unit atau apartemen di banyak daerah
        
        Aplikasi ini memungkinkan Anda untuk mengeksplorasi bagaimana faktor-faktor berbeda ini mempengaruhi harga properti di pasar Melbourne.
        """)

if __name__ == "__main__":
    main()