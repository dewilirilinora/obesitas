import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# KONFIGURASI HALAMAN
# =====================================
st.set_page_config(
    page_title="Sistem Estimasi Risiko Obesitas",
    page_icon="🏥",
    layout="centered"
)

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("obesity_lifestyle.joblib")
le = joblib.load("label_encoder.joblib")

# =====================================
# URUTAN LEVEL OBESITAS (HARUS SAMA DENGAN DATASET)
# =====================================
ORDER = list(le.classes_)  # otomatis ambil dari label encoder

# =====================================
# FUNGSI BMI
# =====================================
def hitung_bmi(weight, height):
    return weight / (height ** 2)

def kategori_bmi(bmi):
    if bmi < 18.5:
        return "Berat Badan Kurang"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesitas"

def bmi_to_level7(bmi):
    if bmi < 18.5:
        return "Insufficient_Weight"
    elif bmi < 25:
        return "Normal_Weight"
    elif bmi < 27.5:
        return "Overweight_Level_I"
    elif bmi < 30:
        return "Overweight_Level_II"
    elif bmi < 35:
        return "Obesity_Type_I"
    elif bmi < 40:
        return "Obesity_Type_II"
    else:
        return "Obesity_Type_III"

# =====================================
# FUNGSI ARAH
# =====================================
def clamp_step(current_label, target_label):
    if current_label not in ORDER or target_label not in ORDER:
        return current_label

    c = ORDER.index(current_label)
    t = ORDER.index(target_label)

    if t > c + 1:
        return ORDER[c + 1]
    if t < c - 1:
        return ORDER[c - 1]
    return ORDER[t]

def arah_perubahan(current_label, lifestyle_label):
    if current_label not in ORDER or lifestyle_label not in ORDER:
        return "Tidak dapat ditentukan"

    c = ORDER.index(current_label)
    l = ORDER.index(lifestyle_label)

    if l < c:
        return "Cenderung TURUN (membaik)"
    elif l > c:
        return "Cenderung NAIK (memburuk)"
    else:
        return "Cenderung STABIL"

# =====================================
# FUNGSI PREDIKSI (PIPELINE VERSION)
# =====================================
def prediksi_lifestyle(data_lifestyle):

    df = pd.DataFrame([data_lifestyle])

    pred_num = model.predict(df)[0]
    pred_label = le.inverse_transform([pred_num])[0]

    proba = model.predict_proba(df)[0]
    confidence = float(np.max(proba))

    return pred_label, confidence

# =====================================
# UI
# =====================================
st.title("🏥 Sistem Estimasi Risiko Obesitas")
st.markdown("Model Machine Learning berbasis Pola Hidup")

st.subheader("Input Data")

gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia (tahun)", 10, 80, 25)

height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.70)
weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0)

fcvc = st.slider("Konsumsi Sayur (1–3)", 1, 3, 2)
ncp = st.slider("Jumlah Makan Utama per Hari", 1, 4, 3)
ch2o = st.slider("Konsumsi Air (1–3)", 1, 3, 2)
faf = st.slider("Aktivitas Fisik (0–3)", 0, 3, 1)
tue = st.slider("Penggunaan Teknologi (0–3)", 0, 3, 1)

family = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori", ["yes", "no"])
caec = st.selectbox("Kebiasaan Ngemil", ["Sometimes", "Frequently", "Always", "no"])
calc = st.selectbox("Konsumsi Alkohol", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Moda Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
scc = st.selectbox("Monitoring Kalori", ["yes", "no"])

# =====================================
# ANALISIS
# =====================================
if st.button("Analisis Risiko"):

    bmi = hitung_bmi(weight, height)
    bmi_cat = kategori_bmi(bmi)
    bmi_level7 = bmi_to_level7(bmi)

    lifestyle_data = {
        "Gender": gender,
        "Age": age,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "family_history_with_overweight": family,
        "FAVC": favc,
        "CAEC": caec,
        "CALC": calc,
        "MTRANS": mtrans,
        "SCC": scc
    }

    pred_label, confidence = prediksi_lifestyle(lifestyle_data)
    pred_label_clean = pred_label.replace("_", " ")

    target_step = clamp_step(bmi_level7, pred_label)
    arah = arah_perubahan(bmi_level7, pred_label)

    st.markdown("---")
    st.header("📋 Hasil Evaluasi")

    st.write(f"**BMI:** {bmi:.2f}")
    st.write(f"**Kategori BMI:** {bmi_cat}")
    st.write(f"**Level Saat Ini:** {bmi_level7.replace('_',' ')}")

    st.write(f"**Prediksi ML (Lifestyle):** {pred_label_clean}")
    st.write(f"**Keyakinan Model:** {confidence:.2%}")

    st.subheader("📈 Estimasi Arah Perubahan")
    st.write(f"**Arah:** {arah}")
    st.write(f"**Target Realistis:** {target_step.replace('_',' ')}")

    if ORDER.index(target_step) < ORDER.index(bmi_level7):
        st.success("Pola hidup mendukung penurunan tingkat obesitas secara bertahap.")
    elif ORDER.index(target_step) > ORDER.index(bmi_level7):
        st.warning("Pola hidup berpotensi meningkatkan risiko ke tingkat lebih tinggi.")
    else:
        st.info("Pola hidup cenderung mempertahankan kondisi saat ini.")
