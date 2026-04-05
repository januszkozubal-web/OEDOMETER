# COLMAN — wariant Streamlit
# Parcia czynne na ścianę oporową Coulomba (odwzorowanie logiki z app.R)
# Autor: Janusz Witalis Kozubal
# Licencja: MIT License
#
# Copyright (c) 2024 Janusz Witalis Kozubal

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon

# --- Funkcje jak w R ---


def kagamma(fi: float, beta: float, delta: float, epsilon: float) -> float:
    fi_rad = np.radians(fi)
    beta_rad = np.radians(beta)
    delta_rad = np.radians(delta)
    epsilon_rad = np.radians(epsilon)
    num = np.cos(fi_rad - beta_rad) ** 2
    den = np.cos(beta_rad + delta_rad)
    inner = np.sin(fi_rad + delta_rad) * np.sin(fi_rad - epsilon_rad) / (
        np.cos(beta_rad + delta_rad) * np.cos(epsilon_rad - beta_rad)
    )
    return num / den / (1 + np.sqrt(inner)) ** 2


def kaq(kagamma_val: float, epsilon: float, beta: float) -> float:
    epsilon_rad = np.radians(epsilon)
    beta_rad = np.radians(beta)
    return kagamma_val / np.cos(epsilon_rad - beta_rad)


def compute_all(
    beta: float,
    epsilon: float,
    load: float,
    h: float,
    fi: float,
    gamma: float,
    delta_factor: float,
):
    delta_value = fi * delta_factor
    kagamma_value = kagamma(fi, beta, delta_value, epsilon)
    kaq_value = kaq(kagamma_value, epsilon, beta)
    delta_rad = np.radians(delta_value)
    beta_rad = -np.radians(beta)
    epsilon_rad = -np.radians(epsilon)

    ax, ay = 0.0, 0.0
    bx = ax + h * np.tan(beta_rad)
    by = ay + h
    l_len = np.sqrt(bx**2 + by**2)
    ea_a = load * kaq_value + kagamma_value * gamma * l_len
    ea_b = load * kaq_value + 0.0

    scale = 0.05
    angle = -beta_rad + delta_rad
    aax = ax + ea_a * np.cos(angle) * scale
    aay = ay + ea_a * np.sin(angle) * scale
    bax = bx + ea_b * np.cos(angle) * scale
    bay = by + ea_b * np.sin(angle) * scale

    df_pressure_x = np.array([ax, aax, bax, bx, ax])
    df_pressure_y = np.array([ay, aay, bay, by, ay])

    ea_ah = ea_a * np.cos(angle)
    ea_av = ea_a * np.sin(angle)
    ea_bh = ea_b * np.cos(angle)
    ea_bv = ea_b * np.sin(angle)

    fh = (ea_bh + ea_ah) * l_len / 2
    fv = (ea_bv + ea_av) * l_len / 2
    hyy = (
        ea_bh * l_len * 0.5 * h + 0.3333 * 0.5 * h * (ea_ah - ea_bh) * l_len
    ) / fh

    vector_position_x = hyy * np.tan(beta_rad)
    vector_position_y = hyy

    cx = bx + h / np.cos(epsilon_rad)
    cy = by - h * np.tan(epsilon_rad)
    dx, dy = cx, 0.0
    fx, fy = bx, by
    gx, gy = cx, cy
    # jak w R: Hx <- Fx; Hy <- Fy + load*scale; Ix <- Gx; Iy <- Gy + load*scale
    hx_r = fx
    hy_r = fy + load * scale
    ix_r = gx
    iy_r = gy + load * scale

    df_figure_x = np.array([ax, bx, cx, dx, ax])
    df_figure_y = np.array([ay, by, cy, dy, ay])
    df_rhomb_x = np.array([fx, gx, ix_r, hx_r, fx])
    df_rhomb_y = np.array([fy, gy, iy_r, hy_r, fy])

    vectors = {
        "x": np.array([aax, bax]),
        "y": np.array([aay, bay]),
        "xend": np.array([ax, bx]),
        "yend": np.array([ay, by]),
    }

    return {
        "kagamma_value": kagamma_value,
        "kaq_value": kaq_value,
        "delta_value": delta_value,
        "ax": ax,
        "ay": ay,
        "bx": bx,
        "by": by,
        "cx": cx,
        "cy": cy,
        "fh": fh,
        "fv": fv,
        "hyy": hyy,
        "ea_a": ea_a,
        "ea_b": ea_b,
        "scale": scale,
        "vector_position_x": vector_position_x,
        "vector_position_y": vector_position_y,
        "df_pressure_x": df_pressure_x,
        "df_pressure_y": df_pressure_y,
        "df_figure_x": df_figure_x,
        "df_figure_y": df_figure_y,
        "df_rhomb_x": df_rhomb_x,
        "df_rhomb_y": df_rhomb_y,
        "vectors": vectors,
        "by_for_text": by,
        "h": h,
    }


def draw_figure(res: dict, gamma: float, fi: float, epsilon: float, beta: float) -> plt.Figure:
    fig, axp = plt.subplots(figsize=(8, 8))

    # Żółty obrys skarpy
    poly_yellow = np.column_stack([res["df_figure_x"], res["df_figure_y"]])
    axp.add_patch(
        MplPolygon(poly_yellow, closed=True, facecolor="yellow", edgecolor="black", linewidth=0.5)
    )

    # Czerwony romb ze wzorem (pionowe kreski — odpowiednik ggpattern)
    poly_red = np.column_stack([res["df_rhomb_x"], res["df_rhomb_y"]])
    axp.add_patch(
        MplPolygon(
            poly_red,
            closed=True,
            facecolor="red",
            edgecolor="darkred",
            linewidth=0.5,
            hatch=r"|||",
        )
    )

    # Ściana AB
    axp.plot(
        [res["ax"], res["bx"]],
        [res["ay"], res["by"]],
        color="black",
        linewidth=3,
        solid_capstyle="round",
    )

    # Szare parcie
    poly_gray = np.column_stack([res["df_pressure_x"], res["df_pressure_y"]])
    axp.add_patch(
        MplPolygon(poly_gray, closed=True, facecolor="gray", edgecolor="gray", alpha=0.5)
    )

    mx = np.mean(res["df_figure_x"])
    my = np.mean(res["df_figure_y"]) / 2
    axp.text(
        mx,
        my,
        (
            f"WYNIKI na 1 mb\nKaq (-): {res['kaq_value']:.3f}\n"
            f"Kagamma (-): {res['kagamma_value']:.3f}\n"
            f"Składowa pozioma siła (kN): {res['fh']:.3f}\n"
            f"Składowa pionowa siła (kN): {res['fv']:.3f}\n"
            f"Wysokość zaczepienia y (m): {res['hyy']:.3f}"
        ),
        fontsize=9,
        ha="center",
        va="center",
    )

    axp.text(res["ax"], res["ay"], f"ea(0,0): {res['ea_a']:.3f}", fontsize=10, ha="left", va="bottom")
    axp.text(res["bx"], res["by"], f"ea(Bx,By): {res['ea_b']:.3f}", fontsize=10, ha="center", va="top")

    parametry = (
        f"Parametry:\nGamma (kN/m^3): {gamma}\n"
        f"Fi (deg): {fi}\n"
        f"Delta (deg): {res['delta_value']:.4g}\n"
        f"Epsilon (deg): {epsilon}\n"
        f"Beta (deg): {beta}\n\n"
        "Autor: JVK ver 1.0 2024"
    )
    axp.text(res["by_for_text"], 1.0, parametry, fontsize=9, ha="center", va="bottom")

    vx = res["vector_position_x"]
    vy = res["vector_position_y"]
    sc = res["scale"]

    # Strzałki wypadkowych (jak geom_segment z arrow)
    arr_h = FancyArrowPatch(
        (vx + res["fh"] * sc, vy),
        (vx, vy),
        arrowstyle="->",
        mutation_scale=12,
        color="black",
    )
    arr_v = FancyArrowPatch(
        (vx, vy + res["fv"] * sc),
        (vx, vy),
        arrowstyle="->",
        mutation_scale=12,
        color="black",
    )
    axp.add_patch(arr_h)
    axp.add_patch(arr_v)

    vdf = res["vectors"]
    for i in range(len(vdf["x"])):
        arr = FancyArrowPatch(
            (vdf["x"][i], vdf["y"][i]),
            (vdf["xend"][i], vdf["yend"][i]),
            arrowstyle="->",
            mutation_scale=12,
            color="black",
        )
        axp.add_patch(arr)

    axp.set_aspect("equal", adjustable="box")
    axp.set_title("Rysunek skarpy z obciążeniem")
    axp.set_xlabel("X")
    axp.set_ylabel("Y")
    axp.grid(False)
    for spine in axp.spines.values():
        spine.set_visible(True)
    axp.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Parcia czynne — skarpa", layout="wide")
    st.title("Animacja parć czynnych dla skarpy z obciążeniem")

    col_side, col_main = st.columns([1, 2])

    with col_side:
        beta = st.slider("Kąt beta - odchylenie od pionu (stopni):", -35, 35, -10, 1)
        epsilon = st.slider("Kąt epsilon - nachylenie naziomu skarpy (stopni):", -20, 20, 10, 1)
        load = st.slider("Obciążenie skarpy (kPa):", 0.0, 20.0, 10.0, 1.0)
        h = st.slider("Wysokość skarpy H (m):", 1.0, 10.0, 5.0, 0.5)
        fi = st.slider("Kąt tarcia wewnętrznego gruntu fi (stopni):", 0, 40, 25, 1)
        gamma = st.slider("Ciężar obj. gruntu - gamma (kN/m^3):", 8.0, 25.0, 20.0, 0.5)
        delta_choice = st.selectbox(
            "Kąt delta tarcia grunt ściana (stopni):",
            options=["0", "1/3 fi", "2/3 fi", "fi"],
            index=1,
        )
        delta_map = {"0": 0.0, "1/3 fi": 1 / 3, "2/3 fi": 2 / 3, "fi": 1.0}
        delta_factor = delta_map[delta_choice]

    res = compute_all(beta, epsilon, load, h, fi, gamma, delta_factor)

    with col_main:
        fig = draw_figure(res, gamma, fi, epsilon, beta)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("COLMAN — Streamlit; logika zgodna z aplikacją Shiny (app.R).")


if __name__ == "__main__":
    main()
