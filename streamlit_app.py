# -*- coding: utf-8 -*-
"""Aplikacja Streamlit — uruchom: streamlit run streamlit_app.py (z katalogu mechanika_labki)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Import modułu projektu (ten sam katalog co ten plik)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PROJEKT_Edometr import oblicz_tabele, rysuj_wykresy

st.set_page_config(page_title="Edometr", layout="wide")

DEFAULT_M = "[0, 0.5, 1, 2, 4, 8, 16, 16, 8, 2, 0]"
DEFAULT_ZI = "[0, 0.05, 0.12, 0.25, 0.35, 0.75, 1.20, 1.20, 1.10, 0.95, 0.11]"
def _parse_list(txt: str):
    try:
        return ast.literal_eval(txt)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Niepoprawna lista: {e}") from e


st.title("Badanie edometryczne")
st.caption("σ' z kg obciążenia, krzywa h(σ'), e(log σ'), moduły Eoed i |Δe/Δlog σ| na odcinkach.")

with st.sidebar:
    st.header("Parametry próbki i aparatury")
    h0 = st.number_input("h₀ [mm]", value=20.0, min_value=0.1)
    d0 = st.number_input("d₀ [mm]", value=75.0, min_value=1.0)
    rho_s = st.number_input("ρₛ (ciężar właściwy stałych cząstek) [-]", value=2.65, format="%.3f")
    w = st.number_input("w (wilgotność) [%]", value=15.0, format="%.2f")
    mm = st.number_input("masa próbki [g]", value=165.0, format="%.2f")
    ramie = st.number_input("ramię (np. 1:10 → 10)", value=10.0, min_value=0.1)

st.subheader("Pomiary (kolejność = ścieżka na wykresie)")
c1, c2, c3 = st.columns(3)
with c1:
    txt_m = st.text_area("m [kg]", value=DEFAULT_M, height=120)
with c2:
    txt_zi = st.text_area("zᵢ (odkształcenie / spęk) [mm]", value=DEFAULT_ZI, height=120)
with c3:
    txt_faza = st.text_area(
        'faza: lista napisów lub wyrażenie, np. ["Obciążanie"]*7 + ["Odciążanie"]*4',
        value='["Obciążanie", "Obciążanie", "Obciążanie", "Obciążanie", "Obciążanie", "Obciążanie", "Obciążanie", "Odciążanie", "Odciążanie", "Odciążanie", "Odciążanie"]',
        height=120,
    )

run = st.button("Przelicz i rysuj", type="primary")

if run or "df_edo" not in st.session_state:
    try:
        m_list = _parse_list(txt_m)
        zi_list = _parse_list(txt_zi)
        faza_list = _parse_list(txt_faza)
        df, stale = oblicz_tabele(
            h0=h0,
            d0=d0,
            rho_s=rho_s,
            w_proc=w,
            masa_probki_g=mm,
            ramie=ramie,
            m_kg=m_list,
            zi_mm=zi_list,
            faza=faza_list,
        )
        st.session_state["df_edo"] = df
        st.session_state["stale_edo"] = stale
        st.session_state["err_edo"] = None
    except Exception as e:
        st.session_state["err_edo"] = str(e)

if st.session_state.get("err_edo"):
    st.error(st.session_state["err_edo"])
elif "df_edo" in st.session_state:
    df = st.session_state["df_edo"]
    stale = st.session_state["stale_edo"]
    st.success(
        f"k = {stale['k_edometr_kPa_per_kg']:.4f} kPa/kg | "
        f"e₀ = {stale['e0']:.4f} | ρ = {stale['rho_g_cm3']:.3f} g/cm³ | "
        f"ρd = {stale['rho_d_g_cm3']:.3f} g/cm³ | V = {stale['V_cm3']:.2f} cm³"
    )
    fig1, fig2 = rysuj_wykresy(df, stale)
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(fig1, use_container_width=True)
    with g2:
        st.pyplot(fig2, use_container_width=True)
    show = df[
        [
            "m",
            "zi",
            "faza",
            "sigma_v",
            "h",
            "e",
            "Eoed_MPa",
            "wskaznik_de_dlog",
        ]
    ].copy()
    st.dataframe(show, use_container_width=True)
    st.download_button(
        "Pobierz CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="edometr_wyniki.csv",
        mime="text/csv",
    )
