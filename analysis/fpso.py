"""FPSO / installation-level aggregation and efficiency analysis."""

import pandas as pd
import numpy as np


def fpso_monthly_production(df: pd.DataFrame, instalacoes: list[str] | None = None) -> pd.DataFrame:
    """
    Aggregate production by installation (FPSO) per month.

    Returns DataFrame with: instalacao, data, oleo_bbl_dia, gas_total,
    agua_bbl_dia, n_pocos, tempo_medio_h, eficiencia
    """
    if "instalacao" not in df.columns:
        return pd.DataFrame()

    filt = df[df["instalacao"].str.strip().str.len() > 0]
    if instalacoes:
        filt = filt[filt["instalacao"].str.strip().isin(instalacoes)]

    if filt.empty:
        return pd.DataFrame()

    monthly = (
        filt.groupby(["instalacao", "data"])
        .agg(
            oleo_bbl_dia=("petroleo_bbl_dia", "sum"),
            gas_total=("gas_total", "sum"),
            agua_bbl_dia=("agua_bbl_dia", "sum"),
            n_pocos=("poco_anp", "nunique"),
            tempo_medio_h=("tempo_producao_h", "mean"),
        )
        .reset_index()
    )

    # Efficiency: hours produced / max hours in month (assuming ~730h/month)
    monthly["eficiencia"] = np.clip(monthly["tempo_medio_h"] / 730 * 100, 0, 100)

    # Production per well
    monthly["oleo_por_poco"] = np.where(
        monthly["n_pocos"] > 0,
        monthly["oleo_bbl_dia"] / monthly["n_pocos"],
        0,
    )

    return monthly.sort_values(["instalacao", "data"]).reset_index(drop=True)


def top_installations(df: pd.DataFrame, n: int = 20) -> list[str]:
    """Return top N installations by average oil production."""
    if "instalacao" not in df.columns or df.empty:
        return []

    df_clean = df[df["instalacao"].str.strip().str.len() > 0].copy()
    avg = (
        df_clean.groupby("instalacao")["petroleo_bbl_dia"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    return avg.index.tolist()


def fpso_summary(df: pd.DataFrame, instalacoes: list[str]) -> pd.DataFrame:
    """
    Summary metrics per FPSO: total production, wells, efficiency, water cut.
    Uses latest month data + historical averages.
    """
    if df.empty or "instalacao" not in df.columns:
        return pd.DataFrame()

    filt = df[df["instalacao"].isin(instalacoes)].copy()
    if filt.empty:
        return pd.DataFrame()

    latest_date = filt["data"].max()
    latest = filt[filt["data"] == latest_date]

    results = []
    for inst in instalacoes:
        inst_all = filt[filt["instalacao"] == inst]
        inst_latest = latest[latest["instalacao"] == inst]

        if inst_all.empty:
            continue

        prod_atual = inst_latest["petroleo_bbl_dia"].sum() if not inst_latest.empty else 0
        n_pocos = inst_latest["poco_anp"].nunique() if not inst_latest.empty else 0
        agua_atual = inst_latest["agua_bbl_dia"].sum() if not inst_latest.empty else 0

        total_liquid = prod_atual + agua_atual
        bsw = (agua_atual / total_liquid * 100) if total_liquid > 0 else 0

        # Average efficiency (compute monthly avg first, then avg across months)
        if "tempo_producao_h" in inst_all.columns:
            monthly_avg_h = inst_all.groupby("data")["tempo_producao_h"].mean()
            eficiencia = min(monthly_avg_h.mean() / 730 * 100, 100)
        else:
            eficiencia = 0

        # First production date
        first_date = inst_all["data"].min()

        results.append({
            "instalacao": inst,
            "prod_atual_bbl_dia": round(prod_atual, 0),
            "n_pocos": n_pocos,
            "bsw_pct": round(bsw, 1),
            "eficiencia_pct": round(eficiencia, 1),
            "primeiro_registro": first_date,
        })

    return pd.DataFrame(results)
