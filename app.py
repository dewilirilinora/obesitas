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
# LOAD MODEL (Pipeline) + Label Encoder
# =====================================
model = joblib.load("obesity_lifestyle.joblib")
le = joblib.load("label_encoder.joblib")

# Urutan kelas dari encoder (paling aman)
ORDER = list(le.classes_)

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
    # mapping BMI umum -> 7 level dataset
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
# ARAH PERUBAHAN (maks 1 level biar realistis)
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
# PREDIKSI (Pipeline) - AUTO FIX MISSING COLUMNS
# =====================================
def prediksi_lifestyle(data_lifestyle):
    df = pd.DataFrame([data_lifestyle])

    # kolom yang diharapkan pipeline
    expected = list(model.named_steps["preprocess"].feature_names_in_)

    # tambahkan kolom yang hilang dengan default aman
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = "no"  # default aman untuk kategorikal yes/no

    # buang kolom yang tidak dipakai model
    df = df[expected]

    pred_num = model.predict(df)[0]
    pred_label = le.inverse_transform([pred_num])[0]
    proba = model.predict_proba(df)[0]
    confidence = float(np.max(proba))

    return pred_label, confidence, missing

# =====================================
# UI
# =====================================
st.title("🏥 Sistem Estimasi Risiko Obesitas")
st.markdown("Model Machine Learning berbasis Pola Hidup (Lifestyle-only) + BMI manual")

st.subheader("Input Data")

# Identitas dasar
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia (tahun)", 10, 80, 25)

# Data fisik (untuk BMI saja, tidak masuk model)
height = st.number_input("Tinggi Badan (meter)", 1.0, 2.5, 1.70)
weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, 70.0)

# Lifestyle (numerik)
fcvc = st.slider("Konsumsi Sayur (FCVC: 1–3)", 1, 3, 2)
ncp  = st.slider("Jumlah Makan Utama per Hari (NCP: 1–4)", 1, 4, 3)
ch2o = st.slider("Konsumsi Air (CH2O: 1–3)", 1, 3, 2)
faf  = st.slider("Aktivitas Fisik (FAF: 0–3)", 0, 3, 1)
tue  = st.slider("Penggunaan Teknologi (TUE: 0–3)", 0, 3, 1)

# Lifestyle (kategorikal)
family = st.selectbox("Riwayat Keluarga Overweight", ["yes", "no"])
favc   = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori (FAVC)", ["yes", "no"])
caec   = st.selectbox("Kebiasaan Ngemil (CAEC)", ["Sometimes", "Frequently", "Always", "no"])
smoke  = st.selectbox("Kebiasaan Merokok (SMOKE)", ["yes", "no"])  # ✅ ini yang sebelumnya hilang
scc    = st.selectbox("Monitoring Kalori (SCC)", ["yes", "no"])
calc   = st.selectbox("Konsumsi Alkohol (CALC)", ["Sometimes", "Frequently", "Always", "no"])
mtrans = st.selectbox("Moda Transportasi (MTRANS)", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# =====================================
# ANALISIS
# =====================================
if st.button("Analisis Risiko"):

    # BMI sekarang (kondisi saat ini)
    bmi = hitung_bmi(weight, height)
    bmi_cat = kategori_bmi(bmi)
    bmi_level7 = bmi_to_level7(bmi)

    # Input ke model (lifestyle-only)
    lifestyle_data = {
        "Gender": gender,
        "Age": float(age),
        "FCVC": float(fcvc),
        "NCP": float(ncp),
        "CH2O": float(ch2o),
        "FAF": float(faf),
        "TUE": float(tue),
        "family_history_with_overweight": family,
        "FAVC": favc,
        "CAEC": caec,
        "SMOKE": smoke,     # ✅ wajib ada
        "SCC": scc,
        "CALC": calc,
        "MTRANS": mtrans
    }

    pred_label, confidence, missing_cols = prediksi_lifestyle(lifestyle_data)
    pred_label_clean = pred_label.replace("_", " ")

    # arah perubahan + target realistis
    target_step = clamp_step(bmi_level7, pred_label)
    arah = arah_perubahan(bmi_level7, pred_label)

    st.markdown("---")
    st.header("📋 Hasil Evaluasi")

    st.write(f"**BMI:** {bmi:.2f}")
    st.write(f"**Kategori BMI:** {bmi_cat}")
    st.write(f"**Level Saat Ini (BMI → 7 level):** {bmi_level7.replace('_',' ')}")

    st.write(f"**Prediksi ML (Lifestyle):** {pred_label_clean}")
    st.write(f"**Keyakinan Model:** {confidence:.2%}")

    st.subheader("📈 Estimasi Arah Perubahan")
    st.write(f"**Arah:** {arah}")
    st.write(f"**Target Realistis (maks 1 tingkat):** {target_step.replace('_',' ')}")

    if missing_cols:
        st.info(f"Catatan: model meminta kolom tambahan yang tidak kamu input, jadi otomatis diisi default: {missing_cols}")

    st.markdown("### 🧾 Kesimpulan")
    if ORDER.index(target_step) < ORDER.index(bmi_level7):
        st.success("Pola hidup kamu mendukung penurunan tingkat obesitas secara bertahap.")
    elif ORDER.index(target_step) > ORDER.index(bmi_level7):
        st.warning("Pola hidup kamu berpotensi meningkatkan risiko ke tingkat lebih tinggi.")
    else:
        st.info("Pola hidup kamu cenderung mempertahankan kondisi saat ini.")
