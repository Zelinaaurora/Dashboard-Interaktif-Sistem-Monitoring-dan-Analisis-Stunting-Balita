#Download & Load Dataset
import kagglehub
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import numpy as np

import os


# Download latest version
path = kagglehub.dataset_download("rendiputra/stunting-balita-detection-121k-rows")

print("Path to dataset files:", path)

# List files to see the actual CSV name
print("Files in dataset:", os.listdir(path))

# Replace with the correct filename from the list
df = pd.read_csv(os.path.join(path, "data_balita.csv"))  # adjust if different
print(df.head())
print(df.columns)



#Data Cleaning Sederhana
# Copy data
df_clean = df.copy()

# Drop missing
df_clean.dropna(inplace=True)

# Pastikan tipe data
df_clean["Umur (bulan)"] = df_clean["Umur (bulan)"].astype(int)

# Normalisasi label stunting
df_clean["Status Gizi"] = df_clean["Status Gizi"].str.lower()



#Setup Streamlit App
st.set_page_config(
    page_title="Dashboard Stunting Balita",
    page_icon="🧒",
    layout="wide"
)

st.title("Dashboard Overview Stunting Balita")
st.markdown("Monitoring & Analisis Stunting Balita")



#Load Data ke Streamlit
@st.cache_data
def load_data():
    return df_clean

df = load_data()

#MENU DASHBOARD
menu = st.sidebar.radio(
    "Menu Dashboard",
    ["📊 Overview Stunting", "🔍 Prediksi Risiko"]
)

if menu == "📊 Overview Stunting":

    #Filter Interaktif
    st.sidebar.header("Filter Data")

    umur_min, umur_max = st.sidebar.slider(
        "Pilih Rentang Umur (bulan)",
        int(df["Umur (bulan)"].min()),
        int(df["Umur (bulan)"].max()),
        (0, 60)
    )

    df = df[
        (df["Umur (bulan)"] >= umur_min) &
        (df["Umur (bulan)"] <= umur_max)
    ]


    #KPI Cards (Ringkasan Utama)
    total_balita = len(df)
    stunting_count = df[df["Status Gizi"].isin(["stunted", "severely stunted"])].shape[0]
    normal_count = total_balita - stunting_count

    col1, col2, col3 = st.columns(3)
    col1.metric("👶 Total Balita", total_balita)
    col2.metric("🚨 Balita Stunting", stunting_count)
    col3.metric("✅ Balita Normal", normal_count)


    #Distribusi Stunting (Pie Chart)
    fig_pie = px.pie(
        df,
        names="Status Gizi",
        title="Distribusi Status Stunting"
    )
    st.plotly_chart(fig_pie, use_container_width=True)


    #Stunting Berdasarkan Jenis Kelamin
    gender_chart = (
        df.groupby(["Jenis Kelamin", "Status Gizi"])
          .size()
          .reset_index(name="Jumlah")
    )

    fig_bar = px.bar(
        gender_chart,
        x="Jenis Kelamin",
        y="Jumlah",
        color="Status Gizi",
        barmode="group",
        title="Status Stunting Berdasarkan Jenis Kelamin"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


    #Distribusi Umur Balita
    fig_age = px.histogram(
        df,
        x="Umur (bulan)",
        color="Status Gizi",
        nbins=20,
        title="Distribusi Umur Balita"
    )
    st.plotly_chart(fig_age, use_container_width=True)





elif menu == "🔍 Prediksi Risiko":

    #LOAD MODEL
    @st.cache_resource
    def load_model():
        model = joblib.load("rf_stunting_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, feature_columns

    model, feature_columns = load_model()


    #SECTION INPUT DATA BALITA
    st.divider()
    st.subheader("🔍 Prediksi Risiko Stunting Balita")

    with st.form("form_prediksi"):
        umur = st.number_input("Umur (bulan)", 0, 60, 24)
        tinggi = st.number_input("Tinggi Badan (cm)", 30.0, 120.0, 80.0)
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

        submit = st.form_submit_button("Prediksi")


    #SAMAKAN INPUT DENGAN FEATURE MODEL
    if submit:
        input_data = {
            "Umur (bulan)": umur,
            "Tinggi Badan (cm)": tinggi,
            "Jenis Kelamin": 1 if jenis_kelamin == "Laki-laki" else 0
        }

        input_df = pd.DataFrame([input_data])

        # pastikan urutan kolom sama
        input_df = input_df[feature_columns]


        #LAKUKAN PREDIKSI & PROBABILITAS
        #prediction = 1: stunting
        #prediction = 0: normal
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]


        #TAMPILKAN HASIL KE USER
        st.subheader("📌 Hasil Prediksi")

        if prediction == 1:
            st.error("⚠️ Balita BERISIKO STUNTING")
            st.write(f"Probabilitas Stunting: **{probability[1]*100:.2f}%**")
        else:
            st.success("✅ Balita TIDAK berisiko stunting")
            st.write(f"Probabilitas Normal: **{probability[0]*100:.2f}%**")


    #FEATURE IMPORTANCE
    st.divider()
    st.subheader("📊 Faktor Paling Berpengaruh")

    importance_df = pd.DataFrame({
        "Fitur": feature_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Fitur",
        orientation="h",
        title="Feature Importance Model Random Forest"
    )

    st.plotly_chart(fig_imp, use_container_width=True)
