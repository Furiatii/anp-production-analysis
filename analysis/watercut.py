"""Water cut (BSW) and Gas-Oil Ratio (GOR) analysis."""

import pandas as pd
import numpy as np


def compute_watercut(df: pd.DataFrame, campos: list[str]) -> pd.DataFrame:
    """
    Compute monthly water cut (BSW) per field.

    Water cut = water / (water + oil) * 100  [%]

    Returns DataFrame with columns: campo, data, watercut, oleo_bbl_dia, agua_bbl_dia
    """
    filt = df[df["campo"].isin(campos)].copy()
    if filt.empty:
        return pd.DataFrame()

    monthly = (
        filt.groupby(["campo", "data"])
        .agg(
            oleo_bbl_dia=("petroleo_bbl_dia", "sum"),
            agua_bbl_dia=("agua_bbl_dia", "sum"),
        )
        .reset_index()
    )

    total_liquid = monthly["oleo_bbl_dia"] + monthly["agua_bbl_dia"]
    monthly["watercut"] = np.where(
        total_liquid > 0,
        monthly["agua_bbl_dia"] / total_liquid * 100,
        0,
    )

    return monthly.sort_values(["campo", "data"]).reset_index(drop=True)


def compute_gor(df: pd.DataFrame, campos: list[str]) -> pd.DataFrame:
    """
    Compute monthly Gas-Oil Ratio (GOR) per field.

    GOR = gas_total (Mm³/dia) / oil (bbl/dia) * 1000  [m³/m³ approx]

    Returns DataFrame with columns: campo, data, gor, oleo_bbl_dia, gas_total
    """
    filt = df[df["campo"].isin(campos)].copy()
    if filt.empty:
        return pd.DataFrame()

    monthly = (
        filt.groupby(["campo", "data"])
        .agg(
            oleo_bbl_dia=("petroleo_bbl_dia", "sum"),
            gas_total=("gas_total", "sum"),
        )
        .reset_index()
    )

    # GOR in m³/m³: gas is in Mm³/dia, oil in bbl/dia
    # Convert oil to m³/dia: 1 bbl = 0.158987 m³
    # GOR = (gas_Mm3_dia * 1000) / (oil_bbl_dia * 0.158987)
    oil_m3 = monthly["oleo_bbl_dia"] * 0.158987
    gas_m3 = monthly["gas_total"] * 1000  # Mm³ -> m³

    monthly["gor"] = np.where(
        oil_m3 > 0,
        gas_m3 / oil_m3,
        0,
    )

    return monthly.sort_values(["campo", "data"]).reset_index(drop=True)


def watercut_summary(wc_df: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per field: current BSW, trend, average."""
    if wc_df.empty:
        return pd.DataFrame()

    results = []
    for campo, grp in wc_df.groupby("campo"):
        grp = grp.sort_values("data")
        current = grp["watercut"].iloc[-1]
        avg = grp["watercut"].mean()

        # Trend: compare last 6 months avg vs previous 6 months
        if len(grp) >= 12:
            recent = grp["watercut"].iloc[-6:].mean()
            older = grp["watercut"].iloc[-12:-6].mean()
            trend = recent - older
        elif len(grp) >= 6:
            recent = grp["watercut"].iloc[-3:].mean()
            older = grp["watercut"].iloc[:-3].mean()
            trend = recent - older
        else:
            trend = 0

        results.append({
            "campo": campo,
            "bsw_atual": round(current, 1),
            "bsw_medio": round(avg, 1),
            "tendencia_pp": round(trend, 1),
            "status": "Subindo" if trend > 2 else ("Estável" if abs(trend) <= 2 else "Caindo"),
        })

    return pd.DataFrame(results)
