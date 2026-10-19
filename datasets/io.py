"""Reading user files (CSV / Excel) without ever modifying the originals."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def read_table(source, filename: str | None = None, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Read a CSV or Excel table from a path, bytes or a file-like object.

    The returned DataFrame is a fresh copy; the source is never written to.
    """
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of {SUPPORTED_EXTENSIONS}.")
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    if suffix == ".csv":
        df = pd.read_csv(source)
    else:
        df = pd.read_excel(source, sheet_name=sheet_name if sheet_name is not None else 0)
    if df.empty:
        raise ValueError("The uploaded file contains no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Column names must be unique after stripping whitespace.")
    return df.copy()


def excel_sheet_names(source) -> list[str]:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    return list(pd.ExcelFile(source).sheet_names)
