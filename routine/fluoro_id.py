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


def classify_unit(udf: pd.DataFrame, zthres, nfluo, exc_lab=[]):
    udf = (
        udf[(~udf["fluo"].isin(exc_lab)) & (udf["beta"] > zthres)]
        .sort_values("beta", ascending=False)
        .reset_index()
    )
    labs = dict()
    for i in range(nfluo):
        try:
            labs["lab{}".format(i)] = udf.loc[i, "fluo"]
            labs["beta{}".format(i)] = udf.loc[i, "beta"]
        except KeyError:
            labs["lab{}".format(i)] = np.nan
            labs["beta{}".format(i)] = np.nan
    return pd.Series(labs)
