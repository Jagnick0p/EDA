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