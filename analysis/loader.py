"""Download and parse ANP production data (oil & gas by well)."""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_URL = (
    "https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/"
    "arquivos/arquivos-producao-de-petroleo-e-gas-natural-por-poco"
)

# Columns we actually use (position-based extraction after parsing)
KEEP_COLS = [
    "estado", "bacia", "poco_anp", "poco_operador", "campo", "operador",
    "contrato", "periodo", "oleo_bbl_dia", "condensado_bbl_dia",
    "petroleo_bbl_dia", "gas_associado", "gas_nao_associado", "gas_total",
    "gas_royalties", "agua_bbl_dia", "instalacao", "tipo_instalacao",
    "tempo_producao_h", "grau_api",
]

# Friendly column names mapped by position (0-indexed)
COL_MAP = {
    0: "estado",
    1: "bacia",
    2: "poco_anp",
    3: "poco_operador",
    4: "campo",
    5: "operador",
    6: "contrato",
    7: "periodo",
    8: "oleo_bbl_dia",
    9: "condensado_bbl_dia",
    10: "petroleo_bbl_dia",
    11: "gas_associado",
    12: "gas_nao_associado",
    13: "gas_total",
    14: "gas_royalties",
    15: "agua_bbl_dia",
    16: "instalacao",
    17: "tipo_instalacao",
    18: "tempo_producao_h",
}

NUMERIC_COLS = [
    "oleo_bbl_dia", "condensado_bbl_dia", "petroleo_bbl_dia",
    "gas_associado", "gas_nao_associado", "gas_total",
    "gas_royalties", "agua_bbl_dia", "tempo_producao_h", "grau_api",
]


def _build_urls(years: list[int]) -> list[tuple[int, str]]:
    """Return (year, url) pairs for the requested years."""
    urls = []
    for y in years:
        if y <= 2022:
            urls.append((y, f"{BASE_URL}/producao-por-poco-{y}.zip"))
        else:
            # 2023+ are monthly ZIPs
            for m in range(1, 13):
                urls.append((y, f"{BASE_URL}/{y}/producao-{m:02d}.zip"))
    return urls


def _download_zip(url: str, cache_path: Path) -> Path | None:
    """Download a ZIP if not cached. Returns path or None on failure."""
    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        return cache_path
    except requests.RequestException:
        return None


def _detect_header_rows(first_lines: list[str]) -> int:
    """Detect how many header/title rows to skip before column headers."""
    for i, line in enumerate(first_lines):
        if line.startswith("Estado;"):
            return i
    # Fallback: look for ANP header pattern in pre-2023 files
    for i, line in enumerate(first_lines):
        if "ANP -" in line or "SIGEP" in line or "BMP -" in line or line.strip() == "":
            continue
        if ";" in line and len(line.split(";")) > 10:
            return i
    return 0


def _parse_csv(file_bytes: bytes, ambiente: str) -> pd.DataFrame:
    """Parse a single CSV file from the ANP ZIP."""
    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    lines = text.splitlines()

    if not lines:
        return pd.DataFrame()

    # Detect where actual column headers start
    skip = _detect_header_rows(lines[:10])

    # Find where data rows start (first row after headers that starts with a state name)
    data_start = skip
    for i in range(skip + 1, min(skip + 5, len(lines))):
        line = lines[i]
        # Sub-header rows contain ANP/Operador or Corte/Volume or are empty
        if not line.strip() or line.startswith(";;") or "Corte;Volume" in line:
            data_start = i + 1
        else:
            break

    if data_start >= len(lines):
        return pd.DataFrame()

    # Parse data rows
    data_lines = lines[data_start:]
    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(";")
        if len(parts) < 15:
            continue
        row = {}
        for pos, col_name in COL_MAP.items():
            if pos < len(parts):
                row[col_name] = parts[pos].strip()
        # Grau API is at position 21 in both formats
        if len(parts) > 21:
            row["grau_api"] = parts[21].strip()
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["ambiente"] = ambiente

    # Convert numeric columns (Brazilian format: comma as decimal)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Parse period (YYYY/MM) into date
    if "periodo" in df.columns:
        df["data"] = pd.to_datetime(df["periodo"], format="%Y/%m", errors="coerce")
        df["ano"] = df["data"].dt.year
        df["mes"] = df["data"].dt.month

    return df


def _extract_ambiente(filename: str) -> str:
    """Detect ambiente (Mar, Pré-Sal, Terra) from filename."""
    lower = filename.lower()
    if "presal" in lower or "pre_sal" in lower or "pré-sal" in lower:
        return "Pré-Sal"
    elif "mar" in lower:
        return "Mar"
    elif "terra" in lower:
        return "Terra"
    return "Desconhecido"


def load_data(
    years: list[int] | None = None,
    ambientes: list[str] | None = None,
    campos: list[str] | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Load ANP production data.

    Parameters
    ----------
    years : list of int, optional
        Years to load. Defaults to [2018, 2019, 2020, 2021, 2022, 2023].
    ambientes : list of str, optional
        Filter by ambiente: "Mar", "Pré-Sal", "Terra". Default: ["Mar", "Pré-Sal"].
    campos : list of str, optional
        Filter by field name (case-insensitive partial match).
    progress_callback : callable, optional
        Called with (current, total) for progress tracking.

    Returns
    -------
    pd.DataFrame
    """
    if years is None:
        years = [2018, 2019, 2020, 2021, 2022, 2023]
    if ambientes is None:
        ambientes = ["Mar", "Pré-Sal"]

    urls = _build_urls(years)
    all_dfs = []

    for idx, (year, url) in enumerate(urls):
        if progress_callback:
            progress_callback(idx, len(urls))

        filename = url.split("/")[-1]
        cache_path = DATA_DIR / f"{year}" / filename
        zip_path = _download_zip(url, cache_path)

        if zip_path is None:
            continue

        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
                for csv_name in csv_files:
                    ambiente = _extract_ambiente(csv_name)
                    if ambiente not in ambientes:
                        continue
                    with zf.open(csv_name) as f:
                        df = _parse_csv(f.read(), ambiente)
                        if not df.empty:
                            all_dfs.append(df)
        except zipfile.BadZipFile:
            continue

    if progress_callback:
        progress_callback(len(urls), len(urls))

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate: Pré-Sal wells appear in both Mar and Pré-Sal files.
    # Keep Pré-Sal label when available (sort so Pré-Sal comes first).
    if "poco_anp" in result.columns and "periodo" in result.columns:
        result["_amb_priority"] = result["ambiente"].map(
            {"Pré-Sal": 0, "Mar": 1, "Terra": 2}
        ).fillna(3)
        result = (
            result.sort_values("_amb_priority")
            .drop_duplicates(subset=["poco_anp", "periodo"], keep="first")
            .drop(columns=["_amb_priority"])
            .reset_index(drop=True)
        )

    # Filter by campos if specified
    if campos and "campo" in result.columns:
        pattern = "|".join(c.upper() for c in campos)
        result = result[result["campo"].str.upper().str.contains(pattern, na=False)]

    return result


def get_available_fields(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique field names."""
    if "campo" not in df.columns or df.empty:
        return []
    return sorted(df["campo"].dropna().unique().tolist())


def get_available_installations(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique installation names."""
    if "instalacao" not in df.columns or df.empty:
        return []
    return sorted(df["instalacao"].dropna().unique().tolist())
