import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ObesityPredict — Dewi Liri Linora",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS CUSTOM
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #f5f3ef;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1a2e !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8e4dc !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label {
    color: #a8a4b8 !important;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Header hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-radius: 20px;
    padding: 48px 52px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(229,57,53,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(100,181,246,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(229,57,53,0.15);
    border: 1px solid rgba(229,57,53,0.35);
    color: #ef9a9a;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 12px 0;
}
.hero-title span {
    color: #ef5350;
}
.hero-sub {
    color: #9e9eb8;
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
    margin: 0;
}
.hero-meta {
    display: flex;
    gap: 24px;
    margin-top: 28px;
    flex-wrap: wrap;
}
.hero-meta-item {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 18px;
}
.hero-meta-item .label {
    font-size: 0.68rem;
    color: #6e6e9e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.hero-meta-item .value {
    font-size: 0.92rem;
    color: #e8e4dc;
    font-weight: 500;
    margin-top: 2px;
}

/* ── Section cards ── */
.section-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid #e8e2d8;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1a1a2e;
    margin: 0 0 6px 0;
}
.section-subtitle {
    font-size: 0.82rem;
    color: #9e9e9e;
    margin: 0 0 24px 0;
    font-weight: 400;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 140px;
    background: #f8f6f2;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #ede8e0;
    text-align: center;
}
.metric-card .m-icon { font-size: 1.6rem; margin-bottom: 6px; }
.metric-card .m-val  { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.metric-card .m-lbl  { font-size: 0.72rem; color: #9e9e9e; text-transform: uppercase;
                        letter-spacing: 0.07em; font-weight: 600; margin-top: 2px; }

/* ── Result banners ── */
.result-banner {
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 16px;
    border-left: 5px solid;
}
.result-banner.green  { background:#e8f5e9; border-color:#43a047; }
.result-banner.yellow { background:#fffde7; border-color:#f9a825; }
.result-banner.orange { background:#fff3e0; border-color:#ef6c00; }
.result-banner.red    { background:#ffebee; border-color:#e53935; }
.result-banner.darkred{ background:#fce4ec; border-color:#b71c1c; }

.result-banner .rb-label {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 4px; opacity: 0.7;
}
.result-banner .rb-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem; color: #1a1a2e; margin: 0 0 6px 0;
}
.result-banner .rb-desc {
    font-size: 0.9rem; color: #555; line-height: 1.6; margin: 0;
}
.confidence-bar-wrap { margin-top: 16px; }
.confidence-bar-wrap .cb-label {
    font-size: 0.78rem; color: #666; margin-bottom: 6px;
    display: flex; justify-content: space-between;
}
.confidence-bar-outer {
    height: 8px; background: rgba(0,0,0,0.08);
    border-radius: 99px; overflow: hidden;
}
.confidence-bar-inner {
    height: 100%; border-radius: 99px;
    transition: width 0.8s ease;
}

/* ── Rekomendasi item ── */
.rek-item {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 14px 0; border-bottom: 1px solid #f0ece5;
}
.rek-item:last-child { border-bottom: none; }
.rek-icon {
    width: 38px; height: 38px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
    background: #f0ece5;
}
.rek-text .rek-head { font-weight: 600; color: #1a1a2e; font-size: 0.9rem; }
.rek-text .rek-body { font-size: 0.82rem; color: #777; margin-top: 2px; line-height: 1.5; }

/* ── BMI gauge ── */
.bmi-wrap {
    background: #f8f6f2; border-radius: 14px; padding: 20px 24px;
    border: 1px solid #ede8e0; margin-bottom: 12px;
}
.bmi-row { display: flex; justify-content: space-between; align-items: center; }
.bmi-num { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
.bmi-tag { font-size: 0.8rem; padding: 4px 12px; border-radius: 20px; font-weight: 600; }

/* ── Divider label ── */
.divider-label {
    display: flex; align-items: center; gap: 12px; margin: 20px 0;
}
.divider-label span { font-size: 0.75rem; font-weight: 600; color: #bbb;
    text-transform: uppercase; letter-spacing: 0.08em; white-space: nowrap; }
.divider-label::before, .divider-label::after {
    content:''; flex: 1; height: 1px; background: #e8e2d8;
}

/* ── Streamlit overrides ── */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #e53935, #b71c1c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] button:hover { opacity: 0.88 !important; }

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
    border: 1.5px solid #e0dbd2 !important;
    background: #faf9f6 !important;
}

/* Footer */
.footer {
    text-align: center; padding: 28px 0 10px;
    font-size: 0.78rem; color: #bbb;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("model_obesitas_xgb.joblib")
    le       = joblib.load("label_encoder.joblib")
    return pipeline, le

try:
    pipeline, le = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
CLASS_INFO = {
    "Insufficient_Weight": {
        "label": "Berat Badan Kurang",
        "color": "yellow",
        "bar_color": "#f9a825",
        "desc": "Berat badan kamu berada di bawah kisaran normal. Kondisi ini dapat memengaruhi imunitas, energi, dan kesehatan tulang.",
        "rekomendasi": [
            ("🥗", "Tambah Asupan Kalori", "Konsumsi makanan bergizi tinggi kalori seperti kacang-kacangan, alpukat, dan biji-bijian utuh."),
            ("💪", "Latihan Kekuatan", "Lakukan latihan beban ringan untuk membantu pembentukan massa otot."),
            ("🩺", "Konsultasi Dokter", "Periksakan ke dokter untuk mengetahui penyebab berat badan kurang."),
        ]
    },
    "Normal_Weight": {
        "label": "Berat Badan Normal",
        "color": "green",
        "bar_color": "#43a047",
        "desc": "Selamat! Berat badanmu berada dalam kisaran normal. Pertahankan pola hidup sehat yang sudah kamu jalani.",
        "rekomendasi": [
            ("🥦", "Jaga Pola Makan", "Lanjutkan konsumsi sayur, buah, dan protein seimbang setiap hari."),
            ("🚶", "Tetap Aktif", "Pertahankan aktivitas fisik minimal 30 menit per hari."),
            ("😴", "Istirahat Cukup", "Tidur 7–8 jam per malam mendukung metabolisme yang sehat."),
        ]
    },
    "Overweight_Level_I": {
        "label": "Kelebihan Berat Badan Tk. I",
        "color": "orange",
        "bar_color": "#ef6c00",
        "desc": "Berat badanmu sedikit di atas normal. Perubahan kecil pada pola makan dan aktivitas fisik sudah cukup efektif.",
        "rekomendasi": [
            ("🚴", "Olahraga Rutin", "Mulai dengan 30–45 menit cardio ringan (jalan cepat, bersepeda) 4–5x seminggu."),
            ("🍽️", "Kurangi Porsi", "Perhatikan ukuran porsi makan, terutama makanan tinggi gula dan lemak jenuh."),
            ("💧", "Cukup Minum Air", "Minum 8 gelas air per hari membantu metabolisme dan mengurangi nafsu makan berlebih."),
        ]
    },
    "Overweight_Level_II": {
        "label": "Kelebihan Berat Badan Tk. II",
        "color": "orange",
        "bar_color": "#e64a19",
        "desc": "Berat badanmu berada di tingkat kelebihan yang lebih signifikan. Diperlukan perubahan gaya hidup yang lebih terstruktur.",
        "rekomendasi": [
            ("🏃", "Program Olahraga", "Ikuti program olahraga terstruktur, minimal 45–60 menit per hari."),
            ("🥗", "Diet Seimbang", "Kurangi makanan olahan, perbanyak sayur, buah, dan protein tanpa lemak."),
            ("👨‍⚕️", "Pantau Kesehatan", "Lakukan pemeriksaan tekanan darah dan gula darah secara rutin."),
        ]
    },
    "Obesity_Type_I": {
        "label": "Obesitas Tipe I",
        "color": "red",
        "bar_color": "#e53935",
        "desc": "Kamu termasuk dalam kategori obesitas tingkat I. Kondisi ini meningkatkan risiko penyakit jantung, diabetes, dan hipertensi.",
        "rekomendasi": [
            ("🩺", "Konsultasi Medis", "Segera konsultasikan dengan dokter atau ahli gizi untuk program penurunan berat badan."),
            ("🏋️", "Latihan Terpandu", "Mulai latihan fisik dengan panduan profesional untuk menghindari cedera."),
            ("🚫", "Hindari Makanan Tinggi Kalori", "Batasi konsumsi makanan cepat saji, minuman manis, dan makanan berlemak tinggi."),
        ]
    },
    "Obesity_Type_II": {
        "label": "Obesitas Tipe II",
        "color": "red",
        "bar_color": "#c62828",
        "desc": "Kategori obesitas tingkat II dengan risiko komplikasi kesehatan yang lebih serius. Penanganan medis sangat dianjurkan.",
        "rekomendasi": [
            ("🏥", "Penanganan Medis", "Segera konsultasi dengan dokter spesialis gizi dan metabolik untuk intervensi yang tepat."),
            ("📋", "Program Diet Ketat", "Ikuti program diet yang diawasi tenaga medis untuk penurunan berat badan yang aman."),
            ("🧘", "Manajemen Stres", "Kelola stres dengan baik karena stres berkontribusi pada peningkatan berat badan."),
        ]
    },
    "Obesity_Type_III": {
        "label": "Obesitas Tipe III (Morbid)",
        "color": "darkred",
        "bar_color": "#b71c1c",
        "desc": "Kategori obesitas tingkat III (morbid obesity). Kondisi ini memerlukan penanganan medis segera dan komprehensif.",
        "rekomendasi": [
            ("🚨", "Penanganan Segera", "Segera konsultasi dokter spesialis — kondisi ini berisiko tinggi terhadap berbagai penyakit serius."),
            ("💊", "Evaluasi Medis Menyeluruh", "Lakukan pemeriksaan lengkap meliputi jantung, gula darah, tekanan darah, dan fungsi organ."),
            ("🏥", "Pertimbangkan Intervensi Klinis", "Dokter mungkin merekomendasikan program klinis intensif atau prosedur medis tertentu."),
        ]
    },
}

def bmi_category(bmi):
    if bmi < 18.5:
        return "Kurang", "#f9a825", "#fff8e1"
    elif bmi < 25.0:
        return "Normal", "#43a047", "#e8f5e9"
    elif bmi < 30.0:
        return "Berlebih", "#ef6c00", "#fff3e0"
    else:
        return "Obesitas", "#e53935", "#ffebee"

def lifestyle_score(favc, caec, smoke, scc, faf, ch2o, calc):
    score = 0
    if favc == "yes": score += 2
    if caec in ["Sometimes","Frequently","Always"]: score += (["Sometimes","Frequently","Always"].index(caec)+1)
    if smoke == "yes": score += 2
    if scc == "no": score += 1
    score += max(0, 2 - faf)
    if ch2o < 2: score += 1
    if calc in ["Sometimes","Frequently"]: score += 1
    max_score = 10
    return min(score, max_score), max_score


# ──────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🔬 Machine Learning · XGBoost · Stratified 10-Fold CV</div>
  <h1 class="hero-title">Prediksi Tingkat<br><span>Risiko Obesitas</span></h1>
  <p class="hero-sub">
    Sistem prediksi berbasis <em>machine learning</em> untuk mendeteksi tingkat risiko obesitas
    secara <em>real-time</em> berdasarkan data gaya hidup dan karakteristik fisik.
  </p>
  <div class="hero-meta">
    <div class="hero-meta-item">
      <div class="label">Algoritma</div>
      <div class="value">XGBoost</div>
    </div>
    <div class="hero-meta-item">
      <div class="label">Validasi</div>
      <div class="value">Stratified 10-Fold CV</div>
    </div>
    <div class="hero-meta-item">
      <div class="label">Akurasi CV</div>
      <div class="value">97.51%</div>
    </div>
    <div class="hero-meta-item">
      <div class="label">Kelas Output</div>
      <div class="value">7 Kategori Obesitas</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ File model tidak ditemukan. Pastikan `model_obesitas_xgb.joblib` dan `label_encoder.joblib` berada di folder yang sama dengan `app.py`.")
    st.stop()


# ──────────────────────────────────────────────────────────────
# SIDEBAR — INPUT DATA
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📝 Input Data Pengguna")
    st.markdown("---")

    st.markdown("**🧬 Data Fisik**")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age    = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25)
    height = st.number_input("Tinggi Badan (m)", min_value=1.40, max_value=2.20,
                              value=1.68, step=0.01, format="%.2f")
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=250.0,
                              value=70.0, step=0.5, format="%.1f")
    family = st.selectbox("Riwayat Keluarga dengan Obesitas",
                           ["yes", "no"], format_func=lambda x: "Ya" if x=="yes" else "Tidak")

    st.markdown("---")
    st.markdown("**🍽️ Pola Makan**")
    favc = st.selectbox("Konsumsi Makanan Tinggi Kalori (FAVC)",
                         ["yes","no"], format_func=lambda x: "Ya" if x=="yes" else "Tidak")
    fcvc = st.slider("Frekuensi Konsumsi Sayur (FCVC)", 1.0, 3.0, 2.0, 0.1)
    ncp  = st.slider("Jumlah Makan Utama per Hari (NCP)", 1.0, 4.0, 3.0, 0.5)
    caec = st.selectbox("Konsumsi Makanan Ringan (CAEC)",
                         ["no","Sometimes","Frequently","Always"],
                         index=1,
                         format_func=lambda x: {
                             "no":"Tidak","Sometimes":"Kadang-kadang",
                             "Frequently":"Sering","Always":"Selalu"
                         }[x])

    st.markdown("---")
    st.markdown("**💧 Kebiasaan & Gaya Hidup**")
    ch2o = st.slider("Konsumsi Air Harian (liter) (CH2O)", 1.0, 3.0, 2.0, 0.1)
    smoke= st.selectbox("Merokok (SMOKE)",
                         ["no","yes"], format_func=lambda x: "Tidak" if x=="no" else "Ya")
    scc  = st.selectbox("Monitoring Konsumsi Kalori (SCC)",
                         ["no","yes"], format_func=lambda x: "Tidak" if x=="no" else "Ya")
    faf  = st.slider("Frekuensi Aktivitas Fisik per Minggu (FAF)", 0.0, 3.0, 1.0, 0.1)
    tue  = st.slider("Durasi Penggunaan Perangkat Teknologi (jam/hari) (TUE)", 0.0, 2.0, 1.0, 0.1)
    calc = st.selectbox("Konsumsi Alkohol (CALC)",
                         ["no","Sometimes","Frequently","Always"],
                         format_func=lambda x: {
                             "no":"Tidak","Sometimes":"Kadang-kadang",
                             "Frequently":"Sering","Always":"Selalu"
                         }[x])
    mtrans = st.selectbox("Transportasi Harian (MTRANS)",
                           ["Public_Transportation","Automobile","Walking",
                            "Motorbike","Bike"],
                           format_func=lambda x: {
                               "Public_Transportation":"Transportasi Umum",
                               "Automobile":"Mobil","Walking":"Jalan Kaki",
                               "Motorbike":"Motor","Bike":"Sepeda"
                           }[x])

    st.markdown("---")
    predict_btn = st.button("🔍 Analisis Risiko Obesitas", use_container_width=True)


# ──────────────────────────────────────────────────────────────
# MAIN AREA — 2 kolom
# ──────────────────────────────────────────────────────────────
col_eval, col_result = st.columns([1, 1], gap="large")

# ── Kolom kiri: Laporan Evaluasi Kesehatan ──
with col_eval:
    st.markdown("""
    <div class="section-card">
      <p class="section-title">📊 Laporan Evaluasi Kesehatan</p>
      <p class="section-subtitle">Ringkasan kondisi fisik dan indikator gaya hidup</p>
    """, unsafe_allow_html=True)

    # BMI
    bmi = weight / (height ** 2)
    bmi_cat, bmi_col, bmi_bg = bmi_category(bmi)

    st.markdown(f"""
    <div class="bmi-wrap">
      <div class="bmi-row">
        <div>
          <div style="font-size:0.75rem;color:#999;text-transform:uppercase;
                      letter-spacing:0.07em;font-weight:600;margin-bottom:4px;">
            Indeks Massa Tubuh (BMI)
          </div>
          <div class="bmi-num">{bmi:.1f} <span style="font-size:1rem;color:#aaa;">kg/m²</span></div>
        </div>
        <div class="bmi-tag" style="background:{bmi_bg};color:{bmi_col};">
          {bmi_cat}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Lifestyle score
    ls, ls_max = lifestyle_score(favc, caec, smoke, scc, faf, ch2o, calc)
    ls_pct = int(ls / ls_max * 100)
    ls_color = "#43a047" if ls <= 3 else ("#ef6c00" if ls <= 6 else "#e53935")
    ls_label = "Baik" if ls <= 3 else ("Perlu Perhatian" if ls <= 6 else "Berisiko Tinggi")

    st.markdown(f"""
    <div class="bmi-wrap" style="margin-top:0;">
      <div class="bmi-row">
        <div>
          <div style="font-size:0.75rem;color:#999;text-transform:uppercase;
                      letter-spacing:0.07em;font-weight:600;margin-bottom:4px;">
            Skor Risiko Gaya Hidup
          </div>
          <div class="bmi-num" style="color:{ls_color};">{ls}<span style="font-size:1rem;color:#aaa;">/{ls_max}</span></div>
        </div>
        <div class="bmi-tag" style="background:{ls_color}22;color:{ls_color};">
          {ls_label}
        </div>
      </div>
      <div style="margin-top:12px;">
        <div class="confidence-bar-outer">
          <div class="confidence-bar-inner"
               style="width:{ls_pct}%;background:{ls_color};"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Ringkasan input
    st.markdown('<div class="divider-label"><span>Ringkasan Input</span></div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
          <div class="m-icon">🧑</div>
          <div class="m-val">{age} thn</div>
          <div class="m-lbl">Usia</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
          <div class="m-icon">⚖️</div>
          <div class="m-val">{weight:.1f} kg</div>
          <div class="m-lbl">Berat Badan</div>
        </div>
        """, unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
          <div class="m-icon">📏</div>
          <div class="m-val">{height:.2f} m</div>
          <div class="m-lbl">Tinggi Badan</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        st.markdown(f"""
        <div class="metric-card">
          <div class="m-icon">🏃</div>
          <div class="m-val">{faf:.1f}x</div>
          <div class="m-lbl">Aktivitas/Minggu</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close section-card


# ── Kolom kanan: Hasil Prediksi ──
with col_result:
    st.markdown("""
    <div class="section-card">
      <p class="section-title">🤖 Hasil Prediksi Model</p>
      <p class="section-subtitle">Output klasifikasi XGBoost & kesimpulan klinis</p>
    """, unsafe_allow_html=True)

    if predict_btn:
        input_df = pd.DataFrame([{
            "Gender": gender, "Age": float(age), "Height": height, "Weight": weight,
            "family_history_with_overweight": family,
            "FAVC": favc, "FCVC": fcvc, "NCP": ncp, "CAEC": caec,
            "SMOKE": smoke, "CH2O": ch2o, "SCC": scc,
            "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
        }])

        with st.spinner("Memproses prediksi..."):
            pred_encoded = pipeline.predict(input_df)[0]
            pred_proba   = pipeline.predict_proba(input_df)[0]
            pred_label   = le.inverse_transform([pred_encoded])[0]
            confidence   = pred_proba[pred_encoded] * 100

        info = CLASS_INFO.get(pred_label, {})
        color      = info.get("color", "yellow")
        bar_color  = info.get("bar_color", "#999")
        label_text = info.get("label", pred_label)
        desc_text  = info.get("desc", "")
        reks       = info.get("rekomendasi", [])

        st.markdown(f"""
        <div class="result-banner {color}">
          <div class="rb-label">Hasil Prediksi Model XGBoost</div>
          <div class="rb-title">{label_text}</div>
          <div class="rb-desc">{desc_text}</div>
          <div class="confidence-bar-wrap">
            <div class="cb-label">
              <span>Tingkat Keyakinan Model</span>
              <span style="font-weight:700;color:{bar_color};">{confidence:.1f}%</span>
            </div>
            <div class="confidence-bar-outer">
              <div class="confidence-bar-inner"
                   style="width:{confidence:.1f}%;background:{bar_color};"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Distribusi probabilitas semua kelas
        st.markdown('<div class="divider-label"><span>Distribusi Probabilitas Semua Kelas</span></div>',
                    unsafe_allow_html=True)

        proba_df = pd.DataFrame({
            "Kategori": le.classes_,
            "Probabilitas (%)": (pred_proba * 100).round(2)
        }).sort_values("Probabilitas (%)", ascending=False)

        for _, row in proba_df.iterrows():
            pct = row["Probabilitas (%)"]
            is_pred = row["Kategori"] == pred_label
            bar_w = pct
            bc = bar_color if is_pred else "#ddd"
            tc = "#1a1a2e" if is_pred else "#999"
            fw = "700" if is_pred else "400"
            st.markdown(f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;
                          font-size:0.78rem;margin-bottom:3px;">
                <span style="color:{tc};font-weight:{fw};">{row['Kategori'].replace('_',' ')}</span>
                <span style="color:{tc};font-weight:{fw};">{pct:.1f}%</span>
              </div>
              <div class="confidence-bar-outer">
                <div class="confidence-bar-inner"
                     style="width:{bar_w:.1f}%;background:{bc};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Rekomendasi
        if reks:
            st.markdown('<div class="divider-label"><span>Kesimpulan Klinis & Rekomendasi</span></div>',
                        unsafe_allow_html=True)
            rek_html = ""
            for icon, head, body in reks:
                rek_html += f"""
                <div class="rek-item">
                  <div class="rek-icon">{icon}</div>
                  <div class="rek-text">
                    <div class="rek-head">{head}</div>
                    <div class="rek-body">{body}</div>
                  </div>
                </div>"""
            st.markdown(rek_html, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#bbb;">
          <div style="font-size:3.5rem;margin-bottom:16px;">🩺</div>
          <div style="font-size:1rem;font-weight:500;color:#999;">
            Isi data di panel kiri, lalu klik<br>
            <strong style="color:#e53935;">Analisis Risiko Obesitas</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Sistem ini hanya untuk keperluan penelitian dan <strong>bukan pengganti diagnosis medis</strong>.<br>
  <span style="color:#d0ccc4;">Dewi Liri Linora · 220401187 · Teknik Informatika · UMRI · 2026</span>
</div>
""", unsafe_allow_html=True)
