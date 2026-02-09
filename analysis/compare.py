"""Compare production ramp-up curves across fields and installations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RampUpMetrics:
    """Metrics for a field's production ramp-up."""
    campo: str
    first_oil_date: pd.Timestamp
    peak_production: float
    months_to_peak: int
    ramp_up_rate: float  # bbl/dia per month on average
    current_production: float
    total_months: int


def compute_rampup(
    df: pd.DataFrame,
    campos: list[str],
    col: str = "petroleo_bbl_dia",
) -> tuple[pd.DataFrame, list[RampUpMetrics]]:
    """
    Compute normalized ramp-up curves and metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    campos : list of str
        Field names to compare.
    col : str
        Production column.

    Returns
    -------
    tuple of (normalized_df, metrics_list)
        normalized_df has columns: campo, months_since_first_oil, production
        metrics_list has one RampUpMetrics per field.
    """
    all_curves = []
    all_metrics = []

    for campo in campos:
        field_df = df[df["campo"].str.upper() == campo.upper()].copy()
        if field_df.empty:
            continue

        # Aggregate monthly production
        monthly = (
            field_df.groupby("data")[col]
            .sum()
            .sort_index()
            .reset_index()
        )

        # First oil = first month with production > 0
        producing = monthly[monthly[col] > 0]
        if producing.empty:
            continue

        first_oil = producing["data"].min()
        monthly["months_since_first_oil"] = (
            (monthly["data"].dt.year - first_oil.year) * 12
            + (monthly["data"].dt.month - first_oil.month)
        )

        # Only keep from first oil onward
        monthly = monthly[monthly["months_since_first_oil"] >= 0].copy()
        monthly["campo"] = campo

        # Metrics
        peak_prod = monthly[col].max()
        peak_row = monthly[monthly[col] == peak_prod].iloc[0]
        months_to_peak = int(peak_row["months_since_first_oil"])
        ramp_rate = peak_prod / max(months_to_peak, 1)
        current_prod = monthly.iloc[-1][col]

        all_curves.append(monthly[["campo", "months_since_first_oil", col]].rename(
            columns={col: "production"}
        ))

        all_metrics.append(RampUpMetrics(
            campo=campo,
            first_oil_date=first_oil,
            peak_production=peak_prod,
            months_to_peak=months_to_peak,
            ramp_up_rate=ramp_rate,
            current_production=current_prod,
            total_months=len(monthly),
        ))

    if not all_curves:
        return pd.DataFrame(), []

    return pd.concat(all_curves, ignore_index=True), all_metrics


def top_fields_by_production(
    df: pd.DataFrame,
    col: str = "petroleo_bbl_dia",
    n: int = 15,
) -> list[str]:
    """Return top N fields by average production."""
    if df.empty or col not in df.columns:
        return []

    avg = (
        df.groupby("campo")[col]
        .mean()
        .sort_values(ascending=False)
        .head(n)
    )
    return avg.index.tolist()


def field_monthly_production(
    df: pd.DataFrame,
    campos: list[str] | None = None,
    col: str = "petroleo_bbl_dia",
) -> pd.DataFrame:
    """
    Get monthly aggregated production per field.

    Returns DataFrame with columns: campo, data, production.
    """
    if campos:
        campos_upper = [c.upper() for c in campos]
        work = df[df["campo"].str.upper().isin(campos_upper)].copy()
    else:
        work = df.copy()

    if work.empty:
        return pd.DataFrame()

    monthly = (
        work.groupby(["campo", "data"])[col]
        .sum()
        .reset_index()
        .rename(columns={col: "production"})
        .sort_values(["campo", "data"])
    )
    return monthly
