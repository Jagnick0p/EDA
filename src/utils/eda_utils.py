import pandas as pd
from pathlib import Path
from pandas.api import types as ptypes
import os
import numpy as np

def missingness(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["name", "dtype", "n_missing", "%missing", "n_non_missing",
                                     "n_unique", "all_missing", "constant", "infinite"])
    rows = []
    for column in df.columns:
        s = df[column]
        dtype = str(s.dtype)
        n_missing = s.isna().sum()
        n_non_missing = df.shape[0] - n_missing
        n_unique = s.nunique(dropna=True)
        pct_missing = round(n_missing/df.shape[0]*100,3)
        complete_missing = n_missing == df.shape[0]
        constant = n_unique == 1
        if not ptypes.is_number(column):
            infinite = 0
        else:
            infinity = np.isinf(s).sum()

        rows.append({
            "name": column,
            "dtype": dtype,
            "n_missing": n_missing,
            "%missing": pct_missing,
            "n_non_missing": n_non_missing,
            "n_unique": n_unique,
            "all_missing": complete_missing,
            "constant": constant,
            "infinite": infinite
        })

    out = pd.DataFrame(rows, columns=["name", "dtype", "n_missing", "%missing", "n_non_missing",
                                      "n_unique", "all_missing", "constant", "infinite"])
    out = out.sort_values(by=["%missing", "name"], ascending=[False, True]).reset_index(drop=True)
    return out

def summarize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(df)
    out_cols = ["name", "count_nonnull", "n_missing", "%missing", "mean", "median",
                "std", "var", "min", "q25", "q75", "max", "iqr", "skew", "kurt", "z_outliers"]

    if n == 0 or len(cols) == 0:
        return pd.DataFrame(columns=out_cols)

    rows = []
    for column in cols:
        s = df[column]
        if not ptypes.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")

        s = s.astype("float64")  # stable numeric ops with NaNs
        count_nonnull = int(s.notna().sum())
        n_missing = n - count_nonnull
        pct_missing = round((n_missing / n) * 100, 1) if n > 0 else 0.0

        mean = float(s.mean())
        median = float(s.median())
        std = float(s.std())
        var = float(s.var())
        min = float(s.min())
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        max = float(s.max())
        iqr = q75 - q25
        skew = float(s.skew())
        kurt = float(s.kurt())
        if std == 0 or count_nonnull < 2 or np.isnan(std):
            z_outliers = 0
        else:
            z = (s-mean)/std
            z_outliers = int((z.abs() > 3 ).sum())

        rows.append({
            "name": column,
            "count_nonnull": count_nonnull,
            "n_missing": n_missing,
            "%missing": pct_missing,
            "mean": mean,
            "median": median,
            "std": std,
            "var": var,
            "min": min,
            "q25": q25,
            "q75": q75,
            "max": max,
            "iqr": iqr,
            "skew": skew,
            "kurt": kurt,
            "z_outliers": z_outliers
        })

    summary = pd.DataFrame(rows, columns=out_cols)
    return summary.sort_values(by="name", ascending=True).reset_index(drop=True)