import os
import pandas as pd
import io
import streamlit as st

# --- Hilfsfunktion zum Laden der Daten (mit Cache) ---
@st.cache_data(show_spinner=False)
def read_excel_cached(source: bytes | str, sheet_name: str | None, is_path: bool) -> pd.DataFrame:
    if is_path:
        return pd.read_excel(source, sheet_name=sheet_name)
    else:
        return pd.read_excel(io.BytesIO(source), sheet_name=sheet_name)


def get_data_from_excel(DEFAULT_PATH):
    data = None
    sheet_selected = None
    if os.path.exists(DEFAULT_PATH):
        source_choice = st.sidebar.radio("Datenquelle", ("Standarddatei", "Upload"), index=0)
        if source_choice == "Standarddatei":
            try:
                xl = pd.ExcelFile(DEFAULT_PATH)
                sheets = xl.sheet_names
                if len(sheets) > 1:
                    sheet_selected = st.sidebar.selectbox("Arbeitsblatt wählen", sheets, index=0)
                else:
                    sheet_selected = sheets[0]
            except Exception:
                sheet_selected = None
            data = read_excel_cached(DEFAULT_PATH, sheet_selected, is_path=True)
        else:
            uploaded_file = st.sidebar.file_uploader("Excel-Datei hochladen", type=["xlsx", "xls"]) 
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                try:
                    xl = pd.ExcelFile(io.BytesIO(bytes_data))
                    sheets = xl.sheet_names
                    if len(sheets) > 1:
                        sheet_selected = st.sidebar.selectbox("Arbeitsblatt wählen", sheets, index=0)
                    else:
                        sheet_selected = sheets[0]
                except Exception:
                    sheet_selected = None
                data = read_excel_cached(bytes_data, sheet_selected, is_path=False)
            else:
                st.sidebar.info("Bitte Datei hochladen oder Standarddatei verwenden.")
    else:
        uploaded_file = st.sidebar.file_uploader("Excel-Datei hochladen", type=["xlsx", "xls"]) 
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            try:
                xl = pd.ExcelFile(io.BytesIO(bytes_data))
                sheets = xl.sheet_names
                if len(sheets) > 1:
                    sheet_selected = st.sidebar.selectbox("Arbeitsblatt wählen", sheets, index=0)
                else:
                    sheet_selected = sheets[0]
            except Exception:
                sheet_selected = None
            data = read_excel_cached(bytes_data, sheet_selected, is_path=False)
        else:
            st.sidebar.info("Keine Standarddatei gefunden – bitte Datei hochladen.")

    return data, sheet_selected