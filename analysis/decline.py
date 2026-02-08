"""Arps decline curve analysis for oil/gas production."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class DeclineResult:
    """Results from decline curve fitting."""
    model: str  # "exponential" or "hyperbolic"
    qi: float   # initial production rate
    di: float   # initial decline rate
    b: float    # Arps b-factor (0 for exponential)
    r_squared: float
    fitted: np.ndarray
    forecast: np.ndarray
    forecast_months: int
    time_actual: np.ndarray
    production_actual: np.ndarray


def _exponential(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    """Exponential decline: q(t) = qi * exp(-di * t)."""
    return qi * np.exp(-di * t)


def _hyperbolic(t: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """Hyperbolic decline: q(t) = qi / (1 + b*di*t)^(1/b)."""
    denominator = 1 + b * di * t
    # Avoid negative values in denominator
    denominator = np.maximum(denominator, 1e-10)
    return qi * np.power(denominator, -1.0 / max(b, 1e-10))


def _r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R² (coefficient of determination)."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_decline_curve(
    df: pd.DataFrame,
    campo: str,
    col: str = "petroleo_bbl_dia",
    forecast_months: int = 24,
) -> DeclineResult | None:
    """
    Fit Arps decline curves to a field's production data.

    Parameters
    ----------
    df : pd.DataFrame
        Production data with 'campo', 'data', and the production column.
    campo : str
        Field name to analyze.
    col : str
        Column with production values (default: petroleo_bbl_dia).
    forecast_months : int
        Number of months to project forward.

    Returns
    -------
    DeclineResult or None if fitting fails.
    """
    field_df = df[df["campo"].str.upper() == campo.upper()].copy()
    if field_df.empty:
        return None

    # Aggregate monthly production by field
    monthly = (
        field_df.groupby("data")[col]
        .sum()
        .sort_index()
        .reset_index()
    )
    monthly = monthly[monthly[col] > 0]

    if len(monthly) < 6:
        return None

    # Find peak production (start decline from peak)
    peak_idx = monthly[col].idxmax()
    decline_data = monthly.loc[peak_idx:].reset_index(drop=True)

    if len(decline_data) < 4:
        return None

    production = decline_data[col].values
    t = np.arange(len(production), dtype=float)
    qi_guess = production[0]

    # Try exponential fit
    exp_result = None
    try:
        popt, _ = curve_fit(
            _exponential, t, production,
            p0=[qi_guess, 0.02],
            bounds=([0, 1e-6], [qi_guess * 3, 1.0]),
            maxfev=5000,
        )
        exp_fitted = _exponential(t, *popt)
        exp_r2 = _r_squared(production, exp_fitted)
        t_forecast = np.arange(len(production), len(production) + forecast_months)
        exp_forecast = _exponential(t_forecast, *popt)
        exp_result = DeclineResult(
            model="exponential", qi=popt[0], di=popt[1], b=0.0,
            r_squared=exp_r2, fitted=exp_fitted,
            forecast=exp_forecast, forecast_months=forecast_months,
            time_actual=t, production_actual=production,
        )
    except (RuntimeError, ValueError):
        pass

    # Try hyperbolic fit
    hyp_result = None
    try:
        popt, _ = curve_fit(
            _hyperbolic, t, production,
            p0=[qi_guess, 0.02, 0.5],
            bounds=([0, 1e-6, 0.01], [qi_guess * 3, 1.0, 2.0]),
            maxfev=5000,
        )
        hyp_fitted = _hyperbolic(t, *popt)
        hyp_r2 = _r_squared(production, hyp_fitted)
        t_forecast = np.arange(len(production), len(production) + forecast_months)
        hyp_forecast = _hyperbolic(t_forecast, *popt)
        hyp_result = DeclineResult(
            model="hyperbolic", qi=popt[0], di=popt[1], b=popt[2],
            r_squared=hyp_r2, fitted=hyp_fitted,
            forecast=hyp_forecast, forecast_months=forecast_months,
            time_actual=t, production_actual=production,
        )
    except (RuntimeError, ValueError):
        pass

    # Return best fit
    if exp_result and hyp_result:
        return hyp_result if hyp_result.r_squared > exp_result.r_squared else exp_result
    return hyp_result or exp_result


def get_decline_dates(df: pd.DataFrame, campo: str) -> pd.DatetimeIndex | None:
    """Get the date range for decline analysis (from peak onward)."""
    field_df = df[df["campo"].str.upper() == campo.upper()]
    if field_df.empty:
        return None

    monthly = (
        field_df.groupby("data")["petroleo_bbl_dia"]
        .sum()
        .sort_index()
    )
    monthly = monthly[monthly > 0]
    if monthly.empty:
        return None

    peak_idx = monthly.idxmax()
    return monthly.loc[peak_idx:].index
