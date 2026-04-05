# -*- coding: utf-8 -*-
# =============================================================================
# PROJEKT_Edometr.py — badanie edometryczne
# Uruchom: python PROJEKT_Edometr.py  → PDF (2 strony) + wydruk tabeli (jak w R_kod_edometr.R)
#
# Co liczy program (σ' z kg obciążenia przez przelicznik k):
#   • krzywa  h = f(σ'),
#   • krzywa  e = f(σ') przy osi σ' w skali logarytmicznej (C_c z NC, C_s z OC / ponowne obciążenie),
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
from matplotlib.lines import Line2D

KATALOG = os.path.dirname(os.path.abspath(__file__))

# Kolory faz (ggplot: niebieski / czerwony / zielony — trzecia faza: ponowne obciążenie, m.in. pod OC)
KOLOR_OBCIAZANIE = "#377EB8"
KOLOR_ODCIAZANIE = "#E41A1C"
KOLOR_PONOWNE_OBCIAZANIE = "#4DAF4A"

KOLORY_FAZ: Dict[str, str] = {
    "Obciążanie": KOLOR_OBCIAZANIE,
    "Odciążanie": KOLOR_ODCIAZANIE,
    "Ponowne obciążanie": KOLOR_PONOWNE_OBCIAZANIE,
}

# Gdy σ′ = 0 w pomiarze, na wykresie półlogarytmicznym e(σ′) podstawiamy σ′ [kPa], żeby uniknąć log(0).
# Wcześniej 0,1 kPa; przyjęto 1 kPa jako dolną sensowną granicę osi log.
SIGMA_LOG_ZAMIAST_ZERA_KPA = 1.0


def fazy_z_m_kg(m_kg: List[float]) -> List[str]:
    """
    Wykrywa fazy po kolejnych masach na równowadze m [kg] (ścieżka czasowa/kolejność kroków).

    Automatyczna maszyna stanów:
      • Obciążanie — dopóki m nie spada (dopuszczalne plateau: m[i] == m[i−1]);
      • Odciążanie — gdy m maleje; dopuszczalne plateau przy min. obciążeniu;
      • Ponowne obciążanie — gdy po fazie odciążenia m znów rośnie;
      • kolejne cykle (znów spadek / wzrost) przełączają między odciążeniem a ponownym obciążeniem.

    Pierwszy punkt jest zawsze „Obciążanie” (początek serii).
    """
    n = len(m_kg)
    if n == 0:
        return []
    if n == 1:
        return ["Obciążanie"]
    m = [float(x) for x in m_kg]
    nazwy = ("Obciążanie", "Odciążanie", "Ponowne obciążanie")
    # state: 0 = pierwsze obciążanie, 1 = odciążanie, 2 = ponowne obciążenie
    state = 0
    out: List[str] = [nazwy[0]] * n
    for i in range(1, n):
        if state == 0:
            if m[i] < m[i - 1]:
                state = 1
        elif state == 1:
            if m[i] > m[i - 1]:
                state = 2
        else:
            if m[i] < m[i - 1]:
                state = 1
        out[i] = nazwy[state]
    return out


def _kolor_dla_fazy(nazwa: str) -> str:
    return KOLORY_FAZ.get(nazwa, "#666666")


def _k_edometr(d0_mm: float, ramie: float) -> float:
    """Przelicznik kg → σ' [kPa] (ramie: stosunek siły na równowadze, np. 1:10 → 10)."""
    return ramie * 9.81 / (np.pi * (d0_mm / 1000.0) ** 2 / 4.0) / 1000.0


def _wolumen_probki_cm3(h0_mm: float, d0_mm: float) -> float:
    """Objętość walca: średnica i wysokość w mm → V w cm³ (jak w R: d0/10, h0/10)."""
    return (np.pi * (d0_mm / 10.0) ** 2 / 4.0) * (h0_mm / 10.0)


def srednia_bez_odstajacych_iqr(values: np.ndarray) -> Tuple[float, int, int]:
    """
    Średnia arytmetyczna po odrzuceniu obserwacji odstających.

    IQR (Interquartile Range / rozstęp międzykwartylowy) = Q3 − Q1. Odstające: poza
    [Q1 − 1,5×IQR, Q3 + 1,5×IQR] (reguła „wąsów” jak w boxplocie). Średnia z wartości nieusuniętych.

    Zwraca: (średnia, liczba wartości użytych do średniej, liczba odrzuconych z próby).
    Przy n < 4 odstających nie odrzuca; przy pustej próbie zwraca (nan, 0, 0).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n_all = int(v.size)
    if n_all == 0:
        return (float("nan"), 0, 0)
    if n_all < 4:
        return (float(np.mean(v)), n_all, 0)
    q1, q3 = np.percentile(v, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        return (float(np.mean(v)), n_all, 0)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (v >= lo) & (v <= hi)
    kept = v[mask]
    n_kept = int(kept.size)
    n_out = n_all - n_kept
    if n_kept == 0:
        return (float(np.median(v)), n_all, n_out)
    return (float(np.mean(kept)), n_kept, n_out)


def iqr_kept_mask(values: np.ndarray) -> np.ndarray:
    """
    Maska tej samej długości co `values`: True tylko tam, gdzie wartość skończona
    weszłaby do średniej z `srednia_bez_odstajacych_iqr` (nie jest odstająca po IQR 1,5×).
    Dla NaN — False. Gdy po IQR nie zostaje żadna wartość (średnia = mediana wszystkich),
    wszystkie skończone → False (żadna pojedyncza nie jest „w średniej arytmetycznej”).
    """
    v = np.asarray(values, dtype=float)
    n = int(v.size)
    out = np.zeros(n, dtype=bool)
    fin = np.isfinite(v)
    if not fin.any():
        return out
    idx_fin = np.flatnonzero(fin)
    vf = v[fin]
    n_all = int(vf.size)
    if n_all == 0:
        return out
    if n_all < 4:
        out[idx_fin] = True
        return out
    q1, q3 = np.percentile(vf, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        out[idx_fin] = True
        return out
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    m = (vf >= lo) & (vf <= hi)
    n_kept = int(np.count_nonzero(m))
    if n_kept == 0:
        return out
    out[idx_fin] = m
    return out


def srednie_odporne(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Średnie Eoed [MPa] oddzielnie: NC (faza „Obciążanie”) i OC („Ponowne obciążenie”).

    |Δe/Δlog σ′|: **C_c** — wyłącznie z odcinków fazy **Obciążanie** (I obciążenie / NC);
    **C_s** — wyłącznie z odcinków fazy **Ponowne obciążenie** (OC, ponowne ściskanie).
    Faza **Odciążanie** nie wchodzi do średnich C_c ani C_s. Wszystko po IQR 1,5×.
    """
    m_nc = df.loc[df["faza"] == "Obciążanie", "Eoed_MPa"].values
    m_oc = df.loc[df["faza"] == "Ponowne obciążanie", "Eoed_MPa"].values
    eoed_nc, n_nc, o_nc = srednia_bez_odstajacych_iqr(m_nc)
    eoed_oc, n_oc, o_oc = srednia_bez_odstajacych_iqr(m_oc)
    m_cc = df.loc[df["faza"] == "Obciążanie", "wskaznik_de_dlog"].values
    m_cs = df.loc[df["faza"] == "Ponowne obciążanie", "wskaznik_de_dlog"].values
    cc, nc, oc = srednia_bez_odstajacych_iqr(m_cc)
    cs, ns, os_ = srednia_bez_odstajacych_iqr(m_cs)
    return {
        "srednia_Eoed_NC_MPa": eoed_nc,
        "srednia_Eoed_NC_n": n_nc,
        "srednia_Eoed_NC_odrzucono": o_nc,
        "srednia_Eoed_OC_MPa": eoed_oc,
        "srednia_Eoed_OC_n": n_oc,
        "srednia_Eoed_OC_odrzucono": o_oc,
        "srednia_Cc": cc,
        "srednia_Cc_n": nc,
        "srednia_Cc_odrzucono": oc,
        "srednia_Cs": cs,
        "srednia_Cs_n": ns,
        "srednia_Cs_odrzucono": os_,
    }


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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Zwraca tabelę pomiarów oraz słownik stałych (k, e₀, ρ, …).

    h0, d0 — wymiary próbki [mm] (wysokość początkowa, średnica).
    rho_s — ρₛ, ciężar właściwy szkieletu  [-], np. Z. Wiłuna.
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
    df["sigma_log"] = np.where(
        df["sigma_v"].values == 0, SIGMA_LOG_ZAMIAST_ZERA_KPA, df["sigma_v"].values
    )

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

    stale: Dict[str, Any] = {
        "k_edometr_kPa_per_kg": k_ed,
        "V_cm3": V,
        "rho_g_cm3": rho,
        "rho_d_g_cm3": rho_d,
        "e0": e0,
    }
    stale.update(srednie_odporne(df))
    return df, stale


def _rysuj_sciezke_faz(ax: plt.Axes, x: np.ndarray, y: np.ndarray, faza: np.ndarray) -> None:
    """Ścieżka łamana z kolorami wg fazy badania."""
    if len(x) < 2:
        return
    for i in range(1, len(x)):
        c = _kolor_dla_fazy(str(faza[i]))
        ax.plot([x[i - 1], x[i]], [y[i - 1], y[i]], color=c, lw=2, solid_capstyle="round")


def _proste_h_od_Eoed(
    ax: plt.Axes,
    sigma_nc: float,
    h_nc: float,
    sigma_oc: float,
    h_oc: float,
    sigma_max: float,
    e_nc_mpa: float,
    e_oc_mpa: float,
) -> None:
    """
    Dwie proste h(σ′) do porównania: nachylenie jak w definicji Eoed w tym pliku,
    dh/dσ′ ≈ −h/(E·1000) (σ′ [kPa], h [mm], E [MPa]).

    NC — od pierwszego pomiaru (początek I etapu, h₀ próbki).
    OC — od pierwszego pomiaru fazy „Ponowne obciążanie” (początek II etapu obciążania).
    """
    def _segment(
        sigma_ref: float,
        h_ref: float,
        e_mpa: float,
        color: str,
        label: str,
    ) -> None:
        if not np.isfinite(e_mpa) or e_mpa <= 0:
            return
        if not np.isfinite(sigma_ref) or not np.isfinite(h_ref):
            return
        if sigma_ref > sigma_max - 1e-12:
            return
        sig = np.array([sigma_ref, sigma_max], dtype=float)
        dh_dsig = -h_ref / (e_mpa * 1000.0)
        h_line = h_ref + dh_dsig * (sig - sigma_ref)
        ax.plot(sig, h_line, "--", color=color, lw=1.8, zorder=4, label=label)

    _segment(
        sigma_nc,
        h_nc,
        e_nc_mpa,
        KOLOR_OBCIAZANIE,
        fr"prosta $E_{{\mathrm{{oed}},\mathrm{{NC}}}}$ = {e_nc_mpa:.2f} MPa",
    )
    _segment(
        sigma_oc,
        h_oc,
        e_oc_mpa,
        KOLOR_PONOWNE_OBCIAZANIE,
        fr"prosta $E_{{\mathrm{{oed}},\mathrm{{OC}}}}$ = {e_oc_mpa:.2f} MPa",
    )


def _proste_e_od_Cc_Cs(
    ax: plt.Axes,
    sigma_cc_ref: float,
    e_cc_ref: float,
    sigma_cs_ref: float,
    e_cs_ref: float,
    sigma_max_plot: float,
    cc: float,
    cs: float,
) -> None:
    """
    Proste na wykresie e(σ′) przy osi σ′ log: nachylenie względem log₁₀ σ′ jak |Δe/Δlog σ′|
    ze średnich C_c (NC) i C_s (OC):

      e = e_ref − C · (log₁₀ σ′ − log₁₀ σ′_ref).

    **C_c** — od pierwszego pomiaru fazy „Obciążanie” do σ′ max w badaniu (I etap / NC).
    **C_s** — od pierwszego pomiaru fazy „Ponowne obciążenie” do σ′ max (II etap obciążenia / OC),
    ta sama postać co wyżej; kolor jak faza OC (nie mylić z odciążaniem).
    """
    def e_na_prostej(sigma: np.ndarray, e_ref: float, sig_ref: float, c: float) -> np.ndarray:
        s = np.maximum(np.asarray(sigma, dtype=float), 1e-15)
        sr = float(max(sig_ref, 1e-15))
        return e_ref - c * (np.log10(s) - np.log10(sr))

    if np.isfinite(cc) and cc > 0 and np.isfinite(sigma_cc_ref) and np.isfinite(e_cc_ref):
        if sigma_cc_ref < sigma_max_plot - 1e-15:
            sig = np.array([sigma_cc_ref, sigma_max_plot], dtype=float)
            ax.plot(
                sig,
                e_na_prostej(sig, e_cc_ref, sigma_cc_ref, cc),
                "--",
                color=KOLOR_OBCIAZANIE,
                lw=1.8,
                zorder=4,
                label=fr"prosta $C_c$ = {cc:.4f}",
            )

    if np.isfinite(cs) and cs > 0 and np.isfinite(sigma_cs_ref) and np.isfinite(e_cs_ref):
        if sigma_cs_ref < sigma_max_plot - 1e-15:
            sig = np.array([sigma_cs_ref, sigma_max_plot], dtype=float)
            ax.plot(
                sig,
                e_na_prostej(sig, e_cs_ref, sigma_cs_ref, cs),
                "--",
                color=KOLOR_PONOWNE_OBCIAZANIE,
                lw=1.8,
                zorder=4,
                label=fr"prosta $C_s$ (OC) = {cs:.4f}",
            )


def _zbuduj_zbiory_in_out_eoed(df: pd.DataFrame) -> Tuple[set, set]:
    """Indeksy wierszy: w / poza średnią Eoed (NC lub OC wg fazy)."""
    faza = df["faza"].values
    eoed = df["Eoed_MPa"].values
    in_set: set = set()
    out_set: set = set()
    sel_nc = faza == "Obciążanie"
    sel_oc = faza == "Ponowne obciążanie"
    kept_nc = iqr_kept_mask(eoed[sel_nc])
    kept_oc = iqr_kept_mask(eoed[sel_oc])
    for k, i in enumerate(np.flatnonzero(sel_nc)):
        if np.isfinite(eoed[i]):
            (in_set if kept_nc[k] else out_set).add(int(i))
    for k, i in enumerate(np.flatnonzero(sel_oc)):
        if np.isfinite(eoed[i]):
            (in_set if kept_oc[k] else out_set).add(int(i))
    return in_set, out_set


def _zbuduj_zbiory_in_out_dlog(df: pd.DataFrame) -> Tuple[set, set]:
    """Indeksy wierszy: w / poza średnią |Δe/Δlog σ′| (C_c z NC, C_s z OC)."""
    faza = df["faza"].values
    w = df["wskaznik_de_dlog"].values
    in_set: set = set()
    out_set: set = set()
    sel_cc = faza == "Obciążanie"
    sel_cs = faza == "Ponowne obciążanie"
    kept_cc = iqr_kept_mask(w[sel_cc])
    kept_cs = iqr_kept_mask(w[sel_cs])
    for k, i in enumerate(np.flatnonzero(sel_cc)):
        if np.isfinite(w[i]):
            (in_set if kept_cc[k] else out_set).add(int(i))
    for k, i in enumerate(np.flatnonzero(sel_cs)):
        if np.isfinite(w[i]):
            (in_set if kept_cs[k] else out_set).add(int(i))
    return in_set, out_set


def _scatter_punkty_faza_iqr(
    ax: plt.Axes,
    xv: np.ndarray,
    yv: np.ndarray,
    faza: np.ndarray,
    in_set: set,
    out_set: set,
) -> None:
    """
    Punkty wg fazy: puste kółka = odrzucone przez IQR przy średniej, pełne = zaliczone;
    pozostałe (inna faza lub brak skończonego wskaźnika) — pełne, bez podziału IQR.
    """
    fazy = ("Obciążanie", "Odciążanie", "Ponowne obciążanie")
    for f in fazy:
        c = KOLORY_FAZ[f]
        idx = np.flatnonzero(faza == f)
        if idx.size == 0:
            continue
        outs = [i for i in idx if i in out_set]
        ins = [i for i in idx if i in in_set]
        nas = [i for i in idx if i not in out_set and i not in in_set]
        lab: Optional[str] = f
        if outs:
            ax.scatter(
                xv[outs],
                yv[outs],
                s=45,
                facecolors="none",
                edgecolors=c,
                linewidths=1.7,
                zorder=5,
                label=lab,
            )
            lab = None
        if ins:
            ax.scatter(xv[ins], yv[ins], s=40, color=c, edgecolors=c, linewidths=0.8, zorder=6, label=lab)
            lab = None
        if nas:
            ax.scatter(xv[nas], yv[nas], s=40, color=c, edgecolors=c, linewidths=0.8, zorder=6, label=lab)


def rysuj_wykresy(
    df: pd.DataFrame,
    stale: Dict[str, Any],
    sigma_step: float = 50.0,
) -> Tuple[plt.Figure, plt.Figure]:
    """Dwa wykresy: h(σ′) liniowo; e(σ′) przy osi σ′ w skali log. Proste: Eoed NC/OC na h(σ′); C_c (NC) i C_s (OC) na e(log σ′)."""
    x = df["sigma_v"].values
    y_h = df["h"].values
    y_e = df["e"].values
    faza = df["faza"].values

    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    _rysuj_sciezke_faz(ax1, x, y_h, faza)
    in_eoed, out_eoed = _zbuduj_zbiory_in_out_eoed(df)
    _scatter_punkty_faza_iqr(ax1, x, y_h, faza, in_eoed, out_eoed)
    leg_h, _ = ax1.get_legend_handles_labels()
    leg_extra_h = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.35",
            markersize=8,
            linestyle="None",
            label="do średniej Eoed (IQR)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.35",
            markerfacecolor="none",
            markersize=8,
            markeredgewidth=1.5,
            linestyle="None",
            label="odstające — Eoed (IQR)",
        ),
    ]
    ax1.legend(handles=list(leg_h) + leg_extra_h, loc="best")
    xmax = float(np.nanmax(x))
    mask_po = df["faza"] == "Ponowne obciążanie"
    if mask_po.any():
        df_po = df.loc[mask_po].iloc[0]
        sigma_oc0 = float(df_po["sigma_v"])
        h_oc0 = float(df_po["h"])
    else:
        sigma_oc0 = float("nan")
        h_oc0 = float("nan")
    _proste_h_od_Eoed(
        ax1,
        float(df["sigma_v"].iloc[0]),
        float(df["h"].iloc[0]),
        sigma_oc0,
        h_oc0,
        xmax,
        float(stale["srednia_Eoed_NC_MPa"]),
        float(stale["srednia_Eoed_OC_MPa"]),
    )
    ax1.set_xlabel("σ′ [kPa]")
    ax1.set_ylabel("h [mm]")
    ax1.set_title("Krzywa edometryczna: h = f(σ′)")
    ax1.grid(True, alpha=0.3)
    ticks = np.arange(0, xmax + sigma_step, sigma_step)
    ticks = ticks[ticks <= xmax + 1e-9]
    ax1.set_xticks(ticks)
    ax1.set_xlim(left=-5, right=xmax)

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    xs = df["sigma_log"].values
    _rysuj_sciezke_faz(ax2, xs, y_e, faza)
    in_dl, out_dl = _zbuduj_zbiory_in_out_dlog(df)
    _scatter_punkty_faza_iqr(ax2, xs, y_e, faza, in_dl, out_dl)

    mask_ob = df["faza"] == "I  Obciążanie"
    mask_po = df["faza"] == "II Obciążanie"
    if mask_ob.any():
        r_cc = df.loc[mask_ob].iloc[0]
        sigma_cc_ref = float(r_cc["sigma_log"])
        e_cc_ref = float(r_cc["e"])
    else:
        sigma_cc_ref = float("nan")
        e_cc_ref = float("nan")
    if mask_po.any():
        r_cs = df.loc[mask_po].iloc[0]
        sigma_cs_ref = float(r_cs["sigma_log"])
        e_cs_ref = float(r_cs["e"])
    else:
        sigma_cs_ref = float("nan")
        e_cs_ref = float("nan")
    sigma_max_plot = float(np.nanmax(xs))
    _proste_e_od_Cc_Cs(
        ax2,
        sigma_cc_ref,
        e_cc_ref,
        sigma_cs_ref,
        e_cs_ref,
        sigma_max_plot,
        float(stale["srednia_Cc"]),
        float(stale["srednia_Cs"]),
    )

    leg_e, _ = ax2.get_legend_handles_labels()
    leg_extra_e = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.35",
            markersize=8,
            linestyle="None",
            label=r"do średniej $C_c$ (NC) / $C_s$ (OC) (IQR)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.35",
            markerfacecolor="none",
            markersize=8,
            markeredgewidth=1.5,
            linestyle="None",
            label=r"odstające — $|\Delta e/\Delta\log\sigma^\prime|$ (IQR)",
        ),
    ]
    ax2.legend(handles=list(leg_e) + leg_extra_e, loc="best")
    ax2.set_xscale("log")
    ax2.set_xlabel("σ′ [kPa] (oś logarytmiczna)")
    ax2.set_ylabel("e [–]")
    ax2.set_title(
        r"e = f(σ′): log $\sigma^\prime$ — $C_c$ (NC), $C_s$ (OC)"
    )
    ax2.grid(True, which="both", alpha=0.3)

    txt1 = (
        f"σ′ z kg: k = {stale['k_edometr_kPa_per_kg']:.4f} kPa/kg | "
        f"e₀ = {stale['e0']:.4f} | "
        f"ρ = {stale['rho_g_cm3']:.3f} g/cm³ | "
        f"ρd = {stale['rho_d_g_cm3']:.3f} g/cm³"
    )
    fig1.suptitle(txt1, fontsize=9, y=0.02)
    txt2 = (
        txt1
        + " | na odcinkach: Eoed [MPa], |Δe/Δlog σ′| — "
        r"$C_c$ (I obciążenie / NC), $C_s$ (II obciążenie / OC)"
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
) -> Tuple[Optional[str], pd.DataFrame, Dict[str, Any], Any]:
    """
    Pełny pipeline: tabela + wykresy + opcjonalnie jeden PDF (2 strony: h(σ′), e(log σ′)).

    Parametry w `d`:
      h0, d0 — wymiary próbki [mm]; rho_s — ρₛ szkieletu (Wiłun); w, mm, ramie;
      m, zi, faza — listy równej długości (fazy można ustalić funkcją `fazy_z_m_kg(m)`).
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


def wydrukuj_podsumowanie(stale: Dict[str, Any], df: pd.DataFrame) -> None:
    print("--- Stałe wstępne ---")
    print(f"k (σ′ z kg obciążenia) [kPa/kg] = {stale['k_edometr_kPa_per_kg']:.6f}")
    print(f"V [cm³] = {stale['V_cm3']:.4f}")
    print(f"ρ [g/cm³] = {stale['rho_g_cm3']:.4f}")
    print(f"ρd [g/cm³] = {stale['rho_d_g_cm3']:.4f}")
    print(f"e₀ = {stale['e0']:.4f}")
    print()
    print("--- Średnie bez odstających (IQR 1,5× na odcinkach z skończonymi wskaźnikami) ---")

    def _fmt(x: Any) -> str:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return "—"
        if not np.isfinite(xf):
            return "—"
        return f"{xf:.6f}"

    print(
        f"Eoed NC [MPa] (I obciążenie): {_fmt(stale['srednia_Eoed_NC_MPa'])} "
        f"(n={stale['srednia_Eoed_NC_n']}, odrzucono {stale['srednia_Eoed_NC_odrzucono']})"
    )
    print(
        f"Eoed OC [MPa] (ponowne obciążenie): {_fmt(stale['srednia_Eoed_OC_MPa'])} "
        f"(n={stale['srednia_Eoed_OC_n']}, odrzucono {stale['srednia_Eoed_OC_odrzucono']})"
    )
    print(
        f"C_c — |Δe/Δlog σ′| (tylko faza I obciążenia / NC, bez ponownego obciążenia): {_fmt(stale['srednia_Cc'])} "
        f"(n={stale['srednia_Cc_n']}, odrzucono {stale['srednia_Cc_odrzucono']})"
    )
    print(
        f"C_s — |Δe/Δlog σ′| (tylko faza ponownego obciążenia / OC): {_fmt(stale['srednia_Cs'])} "
        f"(n={stale['srednia_Cs_n']}, odrzucono {stale['srednia_Cs_odrzucono']})"
    )
    print()
    print(
        "--- Tabela (σ′, faza, Eoed, |Δe/Δlog σ′| — C_c z NC, C_s z OC na wykresie log) ---"
    )
    cols = ["sigma_v", "faza", "Eoed_MPa", "wskaznik_de_dlog"]
    print(df[cols].to_string(index=False))
    print()
    print("--- Pełna tabela ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


def main() -> None:
    m_list = [0, 0.5, 1, 2, 4, 8, 16, 16, 8, 2, 0, 2, 4, 8]
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
        "m": m_list,
        "zi": [0, 0.05, 0.12, 0.25, 0.35, 0.75, 1.20, 1.20, 1.10, 0.95, 0.11, 0.10, 0.11, 0.12],
        "faza": fazy_z_m_kg(m_list),
    }
    path_pdf, df, stale, _ = oblicz_i_rysuj(d, return_figures=False, save_pdf=True)
    print("Zapisano PDF:", path_pdf)
    wydrukuj_podsumowanie(stale, df)

    csv_path = os.path.join(KATALOG, "PROJEKT_Edometr_wyniki.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("Zapisano CSV:", csv_path)


if __name__ == "__main__":
    main()
