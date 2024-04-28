import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression


def fit_spec(spec: xr.DataArray, spec_ref: xr.DataArray):
    lm = LinearRegression(fit_intercept=False)
    lm.fit(spec_ref.transpose("spec", "fluo"), spec.transpose("spec", "unit"))
    return xr.DataArray(
        lm.coef_,
        dims=["unit", "fluo"],
        coords={"unit": spec.coords["unit"], "fluo": spec_ref.coords["fluo"]},
    )


def beta_ztrans(beta: xr.DataArray):
    return xr.apply_ufunc(
        zscore,
        beta,
        input_core_dims=[["unit"]],
        output_core_dims=[["unit"]],
        vectorize=True,
    )


def max_agg_beta(*args):
    return (
        xr.concat([b.expand_dims("agg") for b in args], "agg").max("agg").rename("beta")
    )


def classify_single_unit(udf: pd.DataFrame, zthres):
    udf = udf.set_index("fluo")
    udf = udf[udf["beta"] > zthres].sort_values("beta", ascending=False).reset_index()
    labs = []
    for i in range(len(udf)):
        labs.append(
            pd.Series({"lab": udf.loc[i, "fluo"], "beta": udf.loc[i, "beta"], "ord": i})
        )
    if len(labs) > 0:
        return pd.concat(labs, axis="columns").T.set_index("lab")
    else:
        return pd.DataFrame([{"lab": np.nan, "beta": np.nan, "ord": np.nan}]).set_index(
            "lab"
        )


def classify_units(df: pd.DataFrame, src=None, **kwargs):
    res = (
        df.groupby("unit")
        .apply(classify_single_unit, include_groups=False, **kwargs)
        .reset_index()
    )
    if src is not None:
        res["source"] = src
    return res


def merge_passes(udf: pd.DataFrame, nfluo: int, return_pivot=True):
    udf["source"] = pd.Categorical(udf["source"], ["p1", "p2_raw", "p2_norm"])
    udf = (
        udf.dropna()
        .sort_values(["source", "ord"])
        .drop_duplicates("lab")
        .reset_index(drop=True)
    )
    if return_pivot:
        lab_mg = dict()
        for i in range(nfluo):
            try:
                lab_mg["lab-{}".format(i)] = udf.loc[i, "lab"]
                lab_mg["beta-{}".format(i)] = udf.loc[i, "beta"]
                lab_mg["src-{}".format(i)] = udf.loc[i, "source"]
            except KeyError:
                lab_mg["lab-{}".format(i)] = np.nan
                lab_mg["beta-{}".format(i)] = np.nan
                lab_mg["src-{}".format(i)] = np.nan
        return pd.Series(lab_mg)
    else:
        return udf
