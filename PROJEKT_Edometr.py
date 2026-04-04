# -*- coding: utf-8 -*-
# =============================================================================
# PROJEKT_Edometr.py — badanie edometryczne
# Uruchom: python PROJEKT_Edometr.py  → PDF + wydruk tabeli (jak w R_kod_edometr.R)
#
# Co liczy program (σ' z kg obciążenia przez przelicznik k):
#   • krzywa  h = f(σ'),
#   • krzywa  e = f(σ') przy osi σ' w skali logarytmicznej (nomenklatura C_c / C_s — Wiłun),
#   • moduły edometryczne Eoed [MPa] oraz |Δe/Δlog σ'| na kolejnych odcinkach.
# =============================================================================

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KATALOG = os.path.dirname(os.path.abspath(__file__))

# Kolory faz (spójne z typowym ggplot: niebieski / czerwony)
KOLOR_OBCIAZANIE = "#377EB8"
KOLOR_ODCIAZANIE = "#E41A1C"


def _k_edometr(d0_mm: float, ramie: float) -> float:
    """Przelicznik kg → σ' [kPa] (ramie: stosunek siły na równowadze, np. 1:10 → 10)."""
    return ramie * 9.81 / (np.pi * (d0_mm / 1000.0) ** 2 / 4.0) / 1000.0


def _wolumen_probki_cm3(h0_mm: float, d0_mm: float) -> float:
    """Objętość walca: średnica i wysokość w mm → V w cm³ (jak w R: d0/10, h0/10)."""
    return (np.pi * (d0_mm / 10.0) ** 2 / 4.0) * (h0_mm / 10.0)


def oblicz_tabele(
    h0: float,
    d0: float,
    rho_s: float,
    w_proc: float,
    masa_probki_g: float,
    ramie: float,
    m_kg: List[float],
    zi_mm: List[float],
    faza: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Zwraca tabelę pomiarów oraz słownik stałych (k, e₀, ρ, …).

    h0, d0 — wymiary próbki [mm] (wysokość początkowa, średnica).
    rho_s — ρₛ, ciężar właściwy szkieletu mineralnego (gruntu) [-], wg nomenklatury Z. Wiłuna.
    """
    if not (len(m_kg) == len(zi_mm) == len(faza)):
        raise ValueError("m, zi i faza muszą mieć tę samą długość.")

    k_ed = _k_edometr(d0, ramie)
    V = _wolumen_probki_cm3(h0, d0)
    rho = masa_probki_g / V
    rho_d = rho / (1.0 + w_proc / 100.0)
    e0 = (rho_s / rho_d) - 1.0

    df = pd.DataFrame({"m": m_kg, "zi": zi_mm, "faza": faza})
    df["sigma_v"] = df["m"].values * k_ed
    z1 = df["zi"].iloc[0]
    df["delta_h"] = df["zi"] - z1
    df["h"] = h0 - df["delta_h"]
    df["e"] = e0 - (df["delta_h"] / h0) * (1.0 + e0)
    df["sigma_log"] = np.where(df["sigma_v"].values == 0, 0.1, df["sigma_v"].values)

    sig = df["sigma_v"].values
    h_ = df["h"].values
    e_ = df["e"].values
    sig_log = df["sigma_log"].values

    d_sigma = np.concatenate([[np.nan], np.diff(sig)])
    d_h = np.concatenate([[np.nan], np.diff(h_)])
    d_e = np.concatenate([[np.nan], np.diff(e_)])
    log10_s = np.log10(np.maximum(sig_log, 1e-15))
    d_log_s = np.concatenate([[np.nan], np.diff(log10_s)])

    h_lag = np.roll(h_, 1)
    h_lag[0] = np.nan
    # Eoed [MPa] jak w R: |dσ| / (|dh|/h_poprzednie) / 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(d_h) / h_lag
        e_oed = np.abs(d_sigma / ratio) / 1000.0
    e_oed = np.where(np.isfinite(e_oed), e_oed, np.nan)
    e_oed[0] = np.nan
    mask_zero_dh = (d_h == 0) | (h_lag == 0) | ~np.isfinite(h_lag)
    e_oed = np.where(mask_zero_dh, np.nan, e_oed)

    with np.errstate(divide="ignore", invalid="ignore"):
        wsk = np.abs(d_e / d_log_s)
    wsk = np.where(np.isfinite(wsk), wsk, np.nan)
    wsk[0] = np.nan
    wsk[d_log_s == 0] = np.nan

    df["d_sigma"] = d_sigma
    df["d_h"] = d_h
    df["d_e"] = d_e
    df["d_log_sigma"] = d_log_s
    df["Eoed_MPa"] = e_oed
    df["wskaznik_de_dlog"] = wsk

    stale = {
        "k_edometr_kPa_per_kg": k_ed,
        "V_cm3": V,
        "rho_g_cm3": rho,
        "rho_d_g_cm3": rho_d,
        "e0": e0,
    }
    return df, stale


def _rysuj_sciezke_faz(ax: plt.Axes, x: np.ndarray, y: np.ndarray, faza: np.ndarray) -> None:
    """Ścieżka łamana z kolorami wg fazy (bez strzałek — czytelniejszy rysunek)."""
    if len(x) < 2:
        return
    for i in range(1, len(x)):
        c = KOLOR_OBCIAZANIE if faza[i] == "Obciążanie" else KOLOR_ODCIAZANIE
        ax.plot([x[i - 1], x[i]], [y[i - 1], y[i]], color=c, lw=2, solid_capstyle="round")


def rysuj_wykresy(
    df: pd.DataFrame,
    stale: Dict[str, float],
    sigma_breaks_max: float = 800.0,
    sigma_step: float = 50.0,
) -> Tuple[plt.Figure, plt.Figure]:
    """Dwa wykresy: h(σ') liniowo; e(σ') przy osi σ' w skali log (C_c / C_s wg Wiłuna)."""
    x = df["sigma_v"].values
    y_h = df["h"].values
    y_e = df["e"].values
    faza = df["faza"].values

    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    _rysuj_sciezke_faz(ax1, x, y_h, faza)
    for f, c in [("Obciążanie", KOLOR_OBCIAZANIE), ("Odciążanie", KOLOR_ODCIAZANIE)]:
        mask = df["faza"] == f
        ax1.scatter(x[mask.values], y_h[mask.values], color=c, s=40, zorder=5, label=f)
    ax1.set_xlabel("σ′ [kPa]")
    ax1.set_ylabel("h [mm]")
    ax1.set_title("Krzywa edometryczna: h = f(σ′)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ticks = np.arange(0, sigma_breaks_max + sigma_step, sigma_step)
    ax1.set_xticks(ticks)
    ax1.set_xlim(left=-5, right=max(sigma_breaks_max, np.nanmax(x) * 1.05))

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    xs = df["sigma_log"].values
    _rysuj_sciezke_faz(ax2, xs, y_e, faza)
    for f, c in [("Obciążanie", KOLOR_OBCIAZANIE), ("Odciążanie", KOLOR_ODCIAZANIE)]:
        mask = df["faza"] == f
        ax2.scatter(xs[mask.values], y_e[mask.values], color=c, s=40, zorder=5, label=f)
    ax2.set_xscale("log")
    ax2.set_xlabel("σ′ [kPa] (oś logarytmiczna)")
    ax2.set_ylabel("e [–]")
    ax2.set_title(
        r"e = f(σ′): oś $\sigma^\prime$ log — obciążanie ($C_c$), odciążanie ($C_s$) — Wiłun"
    )
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    txt1 = (
        f"σ′ z kg: k = {stale['k_edometr_kPa_per_kg']:.4f} kPa/kg | "
        f"e₀ = {stale['e0']:.4f} | "
        f"ρ = {stale['rho_g_cm3']:.3f} g/cm³ | "
        f"ρd = {stale['rho_d_g_cm3']:.3f} g/cm³"
    )
    fig1.suptitle(txt1, fontsize=9, y=0.02)
    txt2 = (
        txt1
        + " | na odcinkach: Eoed [MPa], |Δe/Δlog σ′| — przy obciążaniu w kontekście "
        r"$C_c$, przy odciążaniu $C_s$"
    )
    fig2.suptitle(txt2, fontsize=8, y=0.02)
    fig1.subplots_adjust(bottom=0.18)
    fig2.subplots_adjust(bottom=0.18)

    return fig1, fig2


def oblicz_i_rysuj(
    d: Dict[str, Any],
    return_figures: bool = False,
    save_pdf: bool = True,
    pdf_name: str = "PROJEKT_Edometr_wykresy.pdf",
) -> Tuple[Optional[str], pd.DataFrame, Dict[str, float], Any]:
    """
    Pełny pipeline: tabela + wykresy + opcjonalnie jeden PDF (2 strony).

    Parametry w `d`:
      h0, d0 — wymiary próbki [mm]; rho_s — ρₛ szkieletu (Wiłun); w, mm, ramie;
      m, zi, faza — listy równej długości (kroki obciążenia / odciążenia wg `faza`).
    """
    df, stale = oblicz_tabele(
        h0=d["h0"],
        d0=d["d0"],
        rho_s=d["rho_s"],
        w_proc=d["w"],
        masa_probki_g=d["mm"],
        ramie=d["ramie"],
        m_kg=list(d["m"]),
        zi_mm=list(d["zi"]),
        faza=list(d["faza"]),
    )
    fig1, fig2 = rysuj_wykresy(df, stale)

    path_pdf = None
    if save_pdf:
        path_pdf = os.path.join(KATALOG, pdf_name)
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path_pdf) as pdf:
            pdf.savefig(fig1, bbox_inches="tight")
            pdf.savefig(fig2, bbox_inches="tight")

    if not return_figures:
        plt.close(fig1)
        plt.close(fig2)

    if return_figures:
        return path_pdf, df, stale, (fig1, fig2)
    return path_pdf, df, stale, None


def wydrukuj_podsumowanie(stale: Dict[str, float], df: pd.DataFrame) -> None:
    print("--- Stałe wstępne ---")
    print(f"k (σ′ z kg obciążenia) [kPa/kg] = {stale['k_edometr_kPa_per_kg']:.6f}")
    print(f"V [cm³] = {stale['V_cm3']:.4f}")
    print(f"ρ [g/cm³] = {stale['rho_g_cm3']:.4f}")
    print(f"ρd [g/cm³] = {stale['rho_d_g_cm3']:.4f}")
    print(f"e₀ = {stale['e0']:.4f}")
    print()
    print("--- Tabela (σ′, faza, Eoed, |Δe/Δlog σ′| — odniesienie do C_c / C_s na wykresie log) ---")
    cols = ["sigma_v", "faza", "Eoed_MPa", "wskaznik_de_dlog"]
    print(df[cols].to_string(index=False))
    print()
    print("--- Pełna tabela ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


def main() -> None:
    # Liczba punktów (kroków) w fazie obciążenia i odciążenia — musi zgadzać się z listami m, zi.
    N_KROKOW_OBCIAZENIA = 7
    N_KROKOW_ODCIAZENIA = 4

    d: Dict[str, Any] = {
        # h0 — wysokość początkowa próbki [mm]; d0 — średnica próbki (pierścień) [mm]
        "h0": 20.0,
        "d0": 75.0,
        # ρₛ — ciężar właściwy szkieletu mineralnego [–], nomenklatura Z. Wiłuna
        "rho_s": 2.65,
        "w": 15.0,
        "mm": 165.0,
        # Ramię obciążenia (np. 1:10 → 10): przelicznik siły z równowagi → σ′ z m [kg]
        "ramie": 10.0,
        "m": [0, 0.5, 1, 2, 4, 8, 16, 16, 8, 2, 0],
        "zi": [0, 0.05, 0.12, 0.25, 0.35, 0.75, 1.20, 1.20, 1.10, 0.95, 0.11],
        "faza": ["Obciążanie"] * N_KROKOW_OBCIAZENIA + ["Odciążanie"] * N_KROKOW_ODCIAZENIA,
    }
    assert len(d["m"]) == N_KROKOW_OBCIAZENIA + N_KROKOW_ODCIAZENIA
    path_pdf, df, stale, _ = oblicz_i_rysuj(d, return_figures=False, save_pdf=True)
    print("Zapisano PDF:", path_pdf)
    wydrukuj_podsumowanie(stale, df)

    csv_path = os.path.join(KATALOG, "PROJEKT_Edometr_wyniki.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("Zapisano CSV:", csv_path)


if __name__ == "__main__":
    main()
