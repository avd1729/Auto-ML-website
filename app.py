import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

with st.sidebar:
    st.image(
        "https://www.educative.io/v2api/editorpage/5357901372719104/image/5684062662426624")
    st.title("AutoML")
    choice = st.radio(
        "", ["File Upload", "Data Profiling", "ML", "Download model"])

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv", index_col=None)

if choice == "File Upload":
    st.title("Upload Your Data For Modelling")
    file = st.file_uploader("Upload Your Dataset Here ..")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("data.csv", index=None)
        st.dataframe(df)

elif choice == "Data Profiling":
    st.title("Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

elif choice == "ML":
    pass

elif choice == "Download model":
    pass
