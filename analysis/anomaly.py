"""Production anomaly detection."""

import numpy as np
import pandas as pd


def detect_anomalies(
    df: pd.DataFrame,
    campo: str | None = None,
    col: str = "petroleo_bbl_dia",
    drop_threshold: float = 0.20,
    window: int = 6,
) -> pd.DataFrame:
    """
    Detect production anomalies for a field or all fields.

    Anomaly types:
    - "queda": month-over-month drop > threshold
    - "parada": production = 0 after previous production > 0
    - "pico": outlier spike detected via IQR on rolling window

    Parameters
    ----------
    df : pd.DataFrame
        Production data.
    campo : str, optional
        Field name. If None, analyzes all fields.
    col : str
        Production column to analyze.
    drop_threshold : float
        Fractional drop to flag (default 0.20 = 20%).
    window : int
        Rolling window size for IQR outlier detection.

    Returns
    -------
    pd.DataFrame with anomaly flags.
    """
    if campo:
        work = df[df["campo"].str.upper() == campo.upper()]
    else:
        work = df

    if work.empty:
        return pd.DataFrame()

    # Aggregate by field + month
    monthly = (
        work.groupby(["campo", "data"])[col]
        .sum()
        .reset_index()
        .sort_values(["campo", "data"])
    )

    anomalies = []

    for field, group in monthly.groupby("campo"):
        group = group.sort_values("data").reset_index(drop=True)
        prod = group[col].values

        for i in range(1, len(group)):
            anomaly_type = None
            prev = prod[i - 1]
            curr = prod[i]

            # Shutdown: production drops to zero
            if curr == 0 and prev > 0:
                anomaly_type = "parada"

            # Significant drop
            elif prev > 0 and curr > 0:
                change = (curr - prev) / prev
                if change < -drop_threshold:
                    anomaly_type = "queda"

            # IQR spike detection on rolling window
            if anomaly_type is None and i >= window:
                window_data = prod[max(0, i - window):i]
                q1 = np.percentile(window_data, 25)
                q3 = np.percentile(window_data, 75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                if curr > upper_bound and iqr > 0:
                    anomaly_type = "pico"

            if anomaly_type:
                anomalies.append({
                    "campo": field,
                    "data": group.iloc[i]["data"],
                    "producao": curr,
                    "producao_anterior": prev,
                    "variacao_pct": ((curr - prev) / prev * 100) if prev > 0 else None,
                    "tipo": anomaly_type,
                })

    if not anomalies:
        return pd.DataFrame()

    return pd.DataFrame(anomalies)


def anomaly_summary(anomalies_df: pd.DataFrame) -> dict:
    """Summarize anomaly counts by type."""
    if anomalies_df.empty:
        return {"queda": 0, "parada": 0, "pico": 0, "total": 0}

    counts = anomalies_df["tipo"].value_counts().to_dict()
    return {
        "queda": counts.get("queda", 0),
        "parada": counts.get("parada", 0),
        "pico": counts.get("pico", 0),
        "total": len(anomalies_df),
    }
