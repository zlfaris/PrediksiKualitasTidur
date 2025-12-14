import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL & FILE DARI COLAB
model = joblib.load("rf_quality_sleep_model.pkl")
fitur_prediktor = joblib.load("fitur_prediktor.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
bmi_encoder = joblib.load("bmi_encoder.pkl")

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Prediksi Tingkat Kualitas Tidur",
    page_icon="",
    layout="centered"
)

# HEADER
st.markdown(
    """
    <h2 style='text-align:center;'>Prediksi Kualitas Tidur</h2>
    <p style='text-align:center; color:gray;'>
    Berdasarkan Parameter Kesehatan dan Gaya Hidup
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# FORM INPUT USER
with st.form("form_prediksi"):
    st.subheader("Data Pengguna")

    umur = st.number_input(
        "Umur (Tahun)",
        min_value=1,
        max_value=100,
        value=25
    )

    gender_label = st.selectbox(
        "Jenis Kelamin",
        ["Laki-Laki", "Perempuan"]
    )

    aktivitas = st.number_input(
        "Tingkat Aktivitas Fisik (menit/hari)",
        min_value=0,
        max_value=200,
        value=30
    )

    stres = st.number_input(
        "Tingkat Stres (Skala 0–10)",
        min_value=0,
        max_value=10,
        value=5
    )

    heart_rate = st.number_input(
        "Rate Denyut Jantung (bpm)",
        min_value=40,
        max_value=200,
        value=75
    )

    steps = st.number_input(
        "Jumlah Langkah Per Hari",
        min_value=0,
        max_value=30000,
        value=5000
    )

    bmi_label = st.selectbox(
        "Kategori BMI",
        [
            "Berat Badan Normal",
            "Kelebihan Berat Badan",
            "Obesitas"
        ]
    )

    sistolik = st.number_input(
        "Tekanan Darah Sistolik (mmHg)",
        min_value=80,
        max_value=200,
        value=120
    )

    diastolik = st.number_input(
        "Tekanan Darah Diastolik (mmHg)",
        min_value=50,
        max_value=130,
        value=80
    )

    submit = st.form_submit_button("Prediksi Kualitas Tidur")

# PROSES PREDIKSI
if submit:
    # FIX FINAL: Encoding gender (UI → Dataset)
    gender_mapping = {
        "Laki-Laki": "Male",
        "Perempuan": "Female"
    }
    gender_encoded = gender_encoder.transform(
        [gender_mapping[gender_label]]
    )[0]

    # Mapping BMI UI → Dataset
    bmi_mapping = {
        "Berat Badan Normal": "Normal",
        "Berat Badan Normal": "Normal Weight",
        "Kelebihan Berat Badan": "Overweight",
        "Obesitas": "Obese"
    }
    bmi_encoded = bmi_encoder.transform([bmi_mapping[bmi_label]])[0]

    # Susun input sesuai fitur model
    input_data = pd.DataFrame([[
        umur,
        gender_encoded,
        aktivitas,
        stres,
        bmi_encoded,
        heart_rate,
        steps,
        sistolik,
        diastolik
    ]], columns=fitur_prediktor)

    # Prediksi
    prediksi = model.predict(input_data)[0]
    prediksi = round(float(prediksi), 2)

    # OUTPUT HASIL
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Hasil Prediksi")

    st.markdown(
    f"""
    ### Prediksi Tingkat Kualitas Tidur Anda (Skala 1–10):
    <span style="font-weight:600;">{prediksi}/10</span>
    """,
    unsafe_allow_html=True
    )

    # Progress bar
    st.markdown("#### Tingkat Kualitas Tidur")
    st.progress(min(prediksi / 10, 1.0))
    st.caption(f"{int(prediksi * 10)}%")

    # Interpretasi
    if prediksi >= 8:
        st.success("Kualitas Tidur Anda Tergolong Sangat Baik.")
    elif prediksi >= 6:
        st.info("Kualitas Tidur Anda Tergolong Cukup Baik.")
    elif prediksi >= 4:
        st.warning("Kualitas Tidur Anda Tergolong Kurang.")
    else:
        st.error("Kualitas Tidur Anda Tergolong Buruk.")
