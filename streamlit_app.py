# -*- coding: utf-8 -*-
"""Aplikacja Streamlit — uruchom: streamlit run streamlit_app.py."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import UnivariateSpline

# Import modułu projektu (ten sam katalog co ten plik)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from collections import Counter

from PROJEKT_Edometr import (
    SIGMA_LOG_ZAMIAST_ZERA_KPA,
    fazy_z_m_kg,
    oblicz_tabele,
    rysuj_wykresy,
)

st.set_page_config(page_title="Edometr", layout="wide")

# Dymek „?” przy nagłówku sekcji średnich — co to IQR
HELP_IQR = (
    "IQR (rozstęp międzykwartylowy) = Q3 − Q1 — odległość między pierwszym a trzecim kwartylem "
    "zbioru wartości na odcinkach. Jak w diagramie pudełkowym (boxplot): za odstające uznaje się "
    "wartości poza przedziałem [Q1 − 1,5×IQR, Q3 + 1,5×IQR]; dopiero z pozostałych liczona jest średnia. "
    "Gdy jest mniej niż 4 wartości, nic nie odrzucamy."
)

DEFAULT_M = [0, 0.5, 1, 2, 4, 8, 16, 16, 8, 2, 0, 2, 4, 8]
DEFAULT_ZI = [0, 0.12, 0.24, 0.40, 0.54, 0.95, 1.22, 1.22, 1.05, 0.78, 0.60, 0.62, 0.70, 0.78]


def _df_pomiary() -> pd.DataFrame:
    return pd.DataFrame({"m [kg]": DEFAULT_M, "zᵢ [mm]": DEFAULT_ZI})


def _fmt_stale(x: object, nd: int = 4) -> str:
    """Liczba do wyświetlenia albo „—” przy braku / NaN."""
    if x is None:
        return "—"
    try:
        xf = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if math.isnan(xf):
        return "—"
    return f"{xf:.{nd}f}"


def wyznacz_sigma_p_casagrande(df: pd.DataFrame) -> float:
    """
    Automatyczna próba wyznaczenia ciśnienia prekonsolidacji σ′p metodą Casagrande (wykres e–log σ′).

    Działa na fazie „Obciążanie” (I obciążenie / NC). Punkty z σ′ poniżej dolnej granicy osi log
    (jak przy zastępczym σ′ przy zerze w tabeli) są pomijane — spójnie z `SIGMA_LOG_ZAMIAST_ZERA_KPA`.
    """
    # Tylko pierwsze obciążenie (kolumna „faza” == Obciążanie); bez σ′ < 1 kPa (artefakty małych naprężeń / log)
    mask = (df["faza"] == "Obciążanie") & (df["sigma_v"] >= SIGMA_LOG_ZAMIAST_ZERA_KPA)
    df_nc = df.loc[mask, ["sigma_v", "e"]].copy()
    df_nc = df_nc.sort_values("sigma_v").drop_duplicates(subset=["sigma_v"], keep="first")
    if len(df_nc) < 4:
        return float("nan")

    log_s = np.log10(df_nc["sigma_v"].values.astype(float))
    e = df_nc["e"].values.astype(float)

    # 1. Wygładzenie e(log σ′) splajnem — potrzebne do stabilnej pierwszej i drugiej pochodnej (krzywizna).
    # Parametr s > 0: kompromis między dopasowaniem do punktów a gładkością (mniejsze s → bliżej danych).
    spl = UnivariateSpline(log_s, e, s=0.001)

    # Gęsta siatka na odcinku log σ′, żeby wyszukać maksimum krzywizny
    x_fine = np.linspace(log_s.min(), log_s.max(), 500)
    y_fine = spl(x_fine)

    # Pochodne de/d(log σ′) i d²e/d(log σ′)²
    dy = spl.derivative(1)(x_fine)
    ddy = spl.derivative(2)(x_fine)

    # Krzywizna krzywej e vs log σ′: κ = |y″| / (1 + y′²)^(3/2) — klasyczna geometria krzywej y(x)
    curvature = np.abs(ddy) / (1.0 + dy**2) ** 1.5
    idx_max_k = int(np.argmax(curvature))

    x_k = x_fine[idx_max_k]  # log10(σ′) w punkcie maksymalnej krzywizny („kolano”)
    y_k = y_fine[idx_max_k]  # e w tym punkcie
    m_k = dy[idx_max_k]  # nachylenie stycznej de/d(log σ′)

    # 2. Konstrukcja Casagrande: dwusieczna kąta między styczną a poziomem (oś e — poziomo w układzie e vs log σ′)
    angle_tangent = np.arctan(m_k)
    angle_horizontal = 0.0
    angle_bisector = (angle_tangent + angle_horizontal) / 2.0
    m_bisector = np.tan(angle_bisector)

    # 3. Przybliżona „linia dziewiczej ściśliwości” (NC): prosta przez dwa ostatnie punkty obciążania
    # (środek odcinka końcowego, stromego odcinka kompresji normalnej na e–log σ′).
    m_nc_line = (e[-1] - e[-2]) / (log_s[-1] - log_s[-2])
    b_nc_line = e[-1] - m_nc_line * log_s[-1]

    # 4. Przecięcie dwusiecznej z linią NC: y = m_bis·x + b_bis oraz y = m_nc·x + b_nc
    b_bisector = y_k - m_bisector * x_k
    denom = m_bisector - m_nc_line
    if not np.isfinite(denom) or abs(denom) < 1e-14:
        return float("nan")
    log_sigma_p = (b_nc_line - b_bisector) / denom
    sigma_p = 10.0 ** log_sigma_p
    if not np.isfinite(sigma_p) or sigma_p <= 0:
        return float("nan")
    return float(sigma_p)


def _oczysc_pomiary(df: pd.DataFrame) -> pd.DataFrame:
    """Wiersze z  wartościami liczbowymi; kolejność = ścieżka na wykresie."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["m [kg]", "zᵢ [mm]"])
    out = df.copy()
    for c in ("m [kg]", "zᵢ [mm]"):
        if c not in out.columns:
            return pd.DataFrame(columns=["m [kg]", "zᵢ [mm]"])
    out = out.dropna(subset=["m [kg]", "zᵢ [mm]"], how="any")
    return out.reset_index(drop=True)


st.title("Badanie edometryczne")
st.caption(
    "σ′ z kg obciążenia (przelicznik k); krzywa h(σ′); e(σ′) przy osi σ′ logarytmicznej. "
    "Średnie: C_c z odcinków I obciążenia (NC), C_s z odcinków ponownego obciążenia (OC); "
    "Eoed NC / Eoed OC osobno."
)

with st.sidebar:
    st.header("Parametry próbki i aparatury")
    h0 = st.number_input(
        "h₀ [mm] — wysokość początkowa próbki",
        value=20.0,
        min_value=0.1,
        help="Wysokość próbki w pierścieniu przed obciążeniem pomiar suwmiarką.",
    )
    d0 = st.number_input(
        "d₀ [mm] — średnica próbki",
        value=75.0,
        min_value=1.0,
        help="Średnica wewnętrzna pierścienia / próbki pomiar suwmiarką.",
    )
    rho_s = st.number_input(
        "ρₛ [–] — ciężar właściwy szkieletu mineralnego (Wiłun)",
        value=2.65,
        format="%.3f",
        help="Nomenklatura jak u Z. Wiłuna: ciężar właściwy szkieletu gruntowego.",
    )
    w = st.number_input("w (wilgotność) [%]", value=15.0, format="%.2f")
    mm = st.number_input("masa próbki [g]", value=165.0, format="%.2f")
    ramie = st.number_input(
        "Ramię obciążenia (np. 1:10 → 10)",
        value=10.0,
        min_value=0.1,
        help="Stosunek siły na równowadze; stąd σ′ = m·k, k [kPa/kg], wsp. k uwzglednia ramię, powierzchnię i 9.81 m/s².",
    )


st.subheader("Pomiary w kolejności ścieżki na wykresie")
st.caption(
    "Każdy wiersz: obciążenie m [kg] oraz zapis z zegara wartości zᵢ [mm]. "
    "Możesz dodać/usunąć wiersze (+ w tabeli). Fazy (Obciążanie / Odciążanie / "
    "Ponowne obciążanie) program wylicza sam z kolejnych kilogramów — pierwszy spadek m → odciążenie, "
    "pierwszy wzrost po spadku → ponowne obciążenie (plateau przy tym samym m liczy się do bieżącej fazy)."
)
df_edit = st.data_editor(
    _df_pomiary(),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "m [kg]": st.column_config.NumberColumn(
            "m [kg]",
            help="Obciążenie na równowadze [kg]",
            format="%.4f",
            min_value=0.0,
            step=0.01,
        ),
        "zᵢ [mm]": st.column_config.NumberColumn(
            "zᵢ [mm]",
            help="Odczyt z zegara (odkształcenie) [mm]",
            format="%.4f",
            step=0.01,
            min_value=None,
        ),
    },
    key="edo_pomiary_editor",
)

tab = _oczysc_pomiary(df_edit)
n_rows = len(tab)

if n_rows == 0:
    st.warning("Wpisz co najmniej jeden wiersz z m i zᵢ.")
else:
    _preview = fazy_z_m_kg(tab["m [kg]"].astype(float).tolist())
    _cnt = Counter(_preview)
    _parts = [f"{_cnt[k]}× {k}" for k in ("Obciążanie", "Odciążanie", "Ponowne obciążanie") if _cnt[k]]
    st.caption("Wykryte fazy z m [kg]: " + " · ".join(_parts) + f" (łącznie {n_rows} punktów).")

run = st.button("Przelicz i rysuj", type="primary")

if run or "df_edo" not in st.session_state:
    try:
        if n_rows == 0:
            raise ValueError("Brak wierszy pomiarowych (m i zᵢ).")
        m_list = tab["m [kg]"].astype(float).tolist()
        zi_list = tab["zᵢ [mm]"].astype(float).tolist()
        faza_list = fazy_z_m_kg(m_list)
        if len(faza_list) != len(m_list):
            raise ValueError("Wewnętrzny błąd: długość fazy ≠ liczba punktów.")
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

if "df_edo" in st.session_state:
    df = st.session_state["df_edo"]
    stale = st.session_state["stale_edo"]
    st.success(
        f"k = {stale['k_edometr_kPa_per_kg']:.4f} kPa/kg | "
        f"e₀ = {stale['e0']:.4f} | ρ = {stale['rho_g_cm3']:.3f} g/cm³ | "
        f"ρd = {stale['rho_d_g_cm3']:.3f} g/cm³ | V = {stale['V_cm3']:.2f} cm³"
    )
    st.subheader("Średnie bez odstających (IQR 1,5×)", help=HELP_IQR)
    st.caption(
        "Eoed NC / Eoed OC — jak wyżej. C_c i C_s — tak samo rozdzielone: "
        "C_c tylko z fazy „Obciążanie” (NC), C_s tylko z „Ponowne obciążanie” (OC); "
        "faza odciążenia nie wchodzi do C_c ani C_s."
    )
    e1, e2 = st.columns(2)
    with e1:
        st.metric(
            "Eoed NC (śr.) [MPa]",
            _fmt_stale(stale.get("srednia_Eoed_NC_MPa")),
            help=HELP_IQR,
        )
        st.caption(
            f"I obciążenie — n = {stale.get('srednia_Eoed_NC_n', 0)}, "
            f"odrzucono {stale.get('srednia_Eoed_NC_odrzucono', 0)}"
        )
    with e2:
        st.metric(
            "Eoed OC (śr.) [MPa]",
            _fmt_stale(stale.get("srednia_Eoed_OC_MPa")),
            help=HELP_IQR,
        )
        st.caption(
            f"Ponowne obciążenie — n = {stale.get('srednia_Eoed_OC_n', 0)}, "
            f"odrzucono {stale.get('srednia_Eoed_OC_odrzucono', 0)}"
        )
    m2, m3 = st.columns(2)
    with m2:
        st.metric(
            "C_c (NC, śr.) [|Δe/Δlog σ′|]",
            _fmt_stale(stale.get("srednia_Cc")),
            help=HELP_IQR,
        )
        st.caption(
            f"Tylko „Obciążanie” — n = {stale.get('srednia_Cc_n', 0)}, "
            f"odrzucono {stale.get('srednia_Cc_odrzucono', 0)}"
        )
    with m3:
        st.metric(
            "C_s (OC, śr.) [|Δe/Δlog σ′|]",
            _fmt_stale(stale.get("srednia_Cs")),
            help=HELP_IQR,
        )
        st.caption(
            f"Tylko „Ponowne obciążanie” — n = {stale.get('srednia_Cs_n', 0)}, "
            f"odrzucono {stale.get('srednia_Cs_odrzucono', 0)}"
        )
    sigma_p_cas = wyznacz_sigma_p_casagrande(df)
    st.metric(
        "σ′p — Casagrande (auto) [kPa]",
        _fmt_stale(sigma_p_cas),
        help="Szacunek ciśnienia prekonsolidacji z maksimum krzywizny krzywej e–log σ′ (I obciążenie); "
        "dwusieczna + prosta NC z końcowego odcinka — wynik orientacyjny.",
    )
    fig1, fig2 = rysuj_wykresy(df, stale)
    st.subheader("Wykresy: h(σ′) oraz e(log σ′)")
    g1, g2 = st.columns(2)
    with g1:
        st.pyplot(fig1, width="stretch", clear_figure=True)
    with g2:
        st.pyplot(fig2, width="stretch", clear_figure=True)
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
    show = show.rename(
        columns={
            "wskaznik_de_dlog": "|Δe/Δlog σ′| (NC→C_c, OC→C_s; odciążanie bez średniej C)",
        }
    )
    st.dataframe(show, use_container_width=True)
    st.download_button(
        "Pobierz CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="edometr_wyniki.csv",
        mime="text/csv",
    )
